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
    batch_size=4
):
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
    s = torch.log(torch.linalg.svdvals(q.float()).abs() + 1e-9)  # avoid -inf
    
    # avoid large drops in negative singular values from causing a larger h_dim to be predicted
    # do so by multiplying by the sign of the first number -> multiplicative gap remains negative
    count = torch.argmax(
        torch.where(s[:-1] >= 0, 1, -1) * (s[:-1] - s[1:])
    ).item() + 1

    print(f"Hidden Dim: {count}")
    return count, s
