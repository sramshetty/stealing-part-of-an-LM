from tqdm import tqdm
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler


#######################################################################################################################
# Paper Section 4 Methods
#######################################################################################################################
def get_q(
    llama,
    dataset,
    text_key: str,
    n: int = 5000,
    batch_size: int = 1,
    logit_fn: Callable[[nn.Module, str], nn.Tensor] = None
):
    """
    Args:
        llama: llama model
        dataset: dataset to use for random prompt sampling
        text_key: key in dataset that corresponds to the prompts
        n (optional): the number of samples to produce logits for
        batch_size (optional): batch size for model inference
        logit_fn: method to use to compute logit vectors
    
    Returns:
        q: matrix of logit vectors with size (n, l)
    """
    q = torch.zeros(n, llama.tokenizer.n_words)

    assert batch_size == 1, "Currently, only works with batch size of 1"

    random_sampler = RandomSampler(dataset, num_samples=n, generator=torch.Generator(device="cuda"))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=random_sampler,
        generator=torch.Generator(device="cuda")
    )

    if logit_fn:
        for i, batch in enumerate(tqdm(dataloader)):
            prompt = batch[text_key][0]
            # add logits to q as we go
            logits = logit_fn(llama, prompt)
            q[i] = logits.to("cpu")
    else:
        direct_logits(
            llama,
            q,
            dataloader,
            text_key,
            batch_size
        )
    
    return q


@torch.inference_mode()
def direct_logits(
    llama,
    q,
    dataloader,
    text_key: str,
    batch_size: int = 1,
):
    """
    Args:
        llama: llama model
        q: empty matrix of size (n, l) 
        dataloader: dataloader that contains prompts as samples
        text_key: key in dataset that corresponds to the prompts
        batch_size (optional): batch size for model inference
    
    Returns:
        q: matrix of logit vectors with size (n, l)
    """
    for i, batch in enumerate(tqdm(dataloader)):
        prompts = batch[text_key]
        tokens = [llama.tokenizer.encode(p, bos=True, eos=False) for p in prompts]

        # pad tokens following llama implementation
        max_prompt_len = max(len(t) for t in tokens)
        prompt_tokens = torch.full(
            (batch_size, max_prompt_len),
            llama.tokenizer.pad_id,
            dtype=torch.long,
            device="cuda"
        )
        for k, t in enumerate(tokens):
            prompt_tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

        # add logits to q as we go
        logits = llama.model(prompt_tokens, 0)[:, -1]
        q[i * batch_size:i * (batch_size + 1)] = logits.to("cpu")
    
    return


def h_dim_extraction(
    q,
    predict_norm: bool = False
):
    """
    Args:
        q: matrix of logit vectors with size (n, l)
        predict_norm (optional): whehther to predict which normalization layer type is used
    
    Returns:
        u: unitary matrix
        s: singular values
        s_dim: log of the absolute singular values
        count: predicted hidden dimension size
    """    
    # compute singular values and prepare them to find the multiplicative gap
    u, s, _ = torch.linalg.svd(q.T.to(torch.float64), full_matrices=False)
    s_dim = torch.log(s.abs())

    # avoid large drops in negative singular values from causing a larger h_dim to be predicted
    # do so by multiplying by the sign of the first number -> multiplicative gap remains negative
    # also the last singular value is 0 so avoid using it for argmax computation
    count = torch.argmax(
        torch.where(s_dim[:-2] >= 0, 1, -1) * (s_dim[:-2] - s_dim[1:-1])
    ).item() + 1
    
    # TODO: Test with other models
    if predict_norm:
        # Detailed in appendix B.2.2
        q = q.to(torch.float16)
        q_sub = q - q.mean(dim=0)
        del q

        s_sub = torch.linalg.svdvals(q_sub.T.to(torch.float64))
        s_sub_dim = torch.log(s_sub.abs())
        count_sub =  torch.argmax(
            torch.where(s_sub_dim[:-2] >= 0, 1, -1) * (s_sub_dim[:-2] - s_sub_dim[1:-1])
        ).item() + 1

        print("Model uses LayerNorm") if count_sub == count - 1 else print("Model uses RMSNorm")

    print(f"Hidden Dim: {count}")
    return u, s, s_dim, count


def layer_extraction(w, u, s, h_dim):
    """
    Args:
        w: model's actual weight matrix for last layer
        u: unitary matrix computed from `h_dim_extraction`
        s: singular values computed from `h_dim_extraction`
        h_dim: model's actual hidden dimension

    Returns:
        pred_w: predicted w
        g: affine transformation such that pred_w@g ~ w
    """
    w, u, s = w.to("cuda"), u[:, :h_dim].to("cuda"), s[:h_dim].to("cuda")
    pred_w = u @ torch.diag(s)
    g = torch.linalg.lstsq(pred_w, w.to(torch.float64)).solution

    return pred_w, g


#######################################################################################################################
# Paper Section 5 Methods
#######################################################################################################################
@torch.inference_mode()
def topk_logit_extraction(llama, prompt):
    """
    Incrementally construct logit vector by using logit biases to promote sets of tokens for each query.
    Subtract the bias from the output logits to find their actual values and repeat for all tokens.
    """
    n_words = llama.tokenizer.n_words
    out_logits = torch.zeros((n_words,))

    tokens = [llama.tokenizer.encode(prompt, bos=True, eos=False)]

    # pad tokens following llama implementation
    max_prompt_len = max(len(t) for t in tokens)
    prompt_tokens = torch.full(
        (1, max_prompt_len),
        llama.tokenizer.pad_id,
        dtype=torch.long,
        device="cuda"
    )
    for k, t in enumerate(tokens):
        prompt_tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

    # mimic a logit api
    bias = 100
    logits = llama.model(prompt_tokens, 0)[:, -1]

    for i in range(0, n_words, 5):
        # act as though we add bias internally just for demonstration
        sampled_logits = logits.clone()
        sampled_logits[:, i:i+5] += bias

        # return topk logits and their tokens
        outputs = torch.topk(sampled_logits, 5)
        top_logits = outputs.values[0] - bias
        top_tokens = outputs.indices[0]
        
        out_logits[top_tokens] = top_logits.half()  # assume that bias pushes ref token to k
    
    return out_logits


def reference_extraction(llama, prompt):
    n_words = llama.tokenizer.n_words
    logits = torch.zeros((n_words,))

    prompt_tokens = [llama.tokenizer.encode(prompt, bos=True, eos=False)]

    # compute our reference token and its logprob
    out_tokens, logprobs = llama.generate(
        prompt_tokens=prompt_tokens,
        max_gen_len=1,
        temperature=0,
        logprobs=True,
        k=1
    )

    ref_token = out_tokens[0][0]

    bias = 100

    for i in range(0, n_words, 4):
        logitbias = {i+k:bias for k in range(4) if i + k != ref_token}
        out_tokens, logprobs = llama.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=1,
            temperature=0,
            logprobs=True,
            logitbias=logitbias,
            k=5
        )

        token_logits = logprobs[0]["values"][0, -1] - (logprobs[0]["values"][0] - bias)
        tokens = logprobs[0]["tokens"][0, :-1]
        logits[tokens] = token_logits[:-1].half()  # assume that bias pushes ref token to k
    
    return logits
