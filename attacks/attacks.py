from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler


@torch.inference_mode()
def h_dim_extraction(
    tokenizer,
    model,
    dataset,
    text_key,
    n,
    batch_size=4,
    predict_norm=False
):
    """
    Args:
        tokenizer: tokenizer to use for prompt encoding
        model: model to find the hidden dimension of
        dataset: dataset to use for random prompt sampling
        text_key: key in dataset that corresponds to the prompts
        n: the number of samples to produce logits for
        batch_size (Optional): batch size for model inference
        predict_norm (Optional): whehther to predict which normalization layer type is used
    
    Returns:
        u: unitary matrix
        s: singular values
        s_dim: log of the absolute singular values
        count: predicted hidden dimension size
    """
    l = model(torch.tensor([[0]]).to("cuda"), 0).size(-1)
    q = torch.zeros(n, l)

    random_sampler = RandomSampler(dataset, num_samples=n, generator=torch.Generator(device="cuda"))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=random_sampler,
        generator=torch.Generator(device="cuda")
    )

    pad_id = tokenizer.pad_id
    for i, batch in enumerate(tqdm(dataloader)):
        prompts = batch[text_key]
        tokens = [tokenizer.encode(p, bos=True, eos=False) for p in prompts]

        # pad tokens following llama implementation
        max_prompt_len = max(len(t) for t in tokens)
        prompt_tokens = torch.full(
            (batch_size, max_prompt_len),
            pad_id,
            dtype=torch.long,
            device="cuda"
        )
        for k, t in enumerate(tokens):
            prompt_tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

        # add logits to q as we go
        logits = model(prompt_tokens, 0)[:, -1]
        q[i * batch_size:i * (batch_size + 1)] = logits.to("cpu")
    
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