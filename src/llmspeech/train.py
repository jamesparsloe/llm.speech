import os
import time
from functools import partial

import click
import torch
import torch.nn.functional as F
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import wandb
from llmspeech.dataset import DATASET_DIR, SNACDataset
from llmspeech.utils import cycle, warmup_then_cosine_decay

from .config import Config
from .model import GPT, build_optimizer


def collate(items, pad_token_id: int):
    input_ids = pad_sequence(
        items,
        batch_first=True,
        padding_value=pad_token_id,
    )
    return input_ids


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--edit", is_flag=True)
def main(config_path: str, edit: bool):
    with open(config_path) as f:
        s = f.read()

        if edit:
            s = click.edit(s)

        config = Config(**yaml.safe_load(s))

    device = "cuda"

    run = wandb.init(project="llm.speech", config=config.model_dump())

    run_dir = os.path.join("./runs", run.id)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        f.write(yaml.dump(config.model_dump()))

    kind = config.kind

    assert kind == "gpt"

    model = GPT(config).to(device)

    weight_decay = config.weight_decay
    lr = config.lr
    betas = config.betas

    optimizer = build_optimizer(model, weight_decay=weight_decay, lr=lr, betas=betas)

    dataset = SNACDataset(
        DATASET_DIR,
        n_text_tokens=config.n_text_tokens,
        codebook_size=config.codebook_size,
        bos_token_id=config.bos_token_id,
        boa_token_id=config.boa_token_id,
        eos_token_id=config.eos_token_id,
    )

    n_items = len(dataset)
    epoch_size = n_items // config.batch_size

    print(f"Dataset consists of {n_items} items. This is {epoch_size} steps per epoch.")

    gradient_accumulation_steps = config.grad_accumulation_steps
    dl = DataLoader(
        dataset,
        batch_size=config.micro_batch_size,
        shuffle=True,
        collate_fn=partial(collate, pad_token_id=config.pad_token_id),
        num_workers=config.micro_batch_size,
    )
    log_every = 10

    min_lr = lr / 10
    steps = config.steps
    checkpoint_every = config.checkpoint_every
    warmup_steps = config.warmup_steps

    batch_size = config.batch_size

    get_lr = partial(
        warmup_then_cosine_decay,
        warmup_steps=warmup_steps,
        steps=steps,
        min_lr=min_lr,
        max_lr=lr,
    )

    amp_dtype = torch.bfloat16

    dl_iter = cycle(dl)

    max_grad_norm = config.max_grad_norm
    step = 0

    while step < steps:
        lr = get_lr(step)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        t1 = time.perf_counter()

        for microstep in range(gradient_accumulation_steps):
            token_ids = next(dl_iter)

            token_ids = token_ids.to(device)
            input_ids, target_ids = token_ids[..., :-1], token_ids[..., 1:].contiguous()

            with torch.amp.autocast(dtype=amp_dtype, device_type="cuda", enabled=True):
                logits = model(input_ids=input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1),
                    ignore_index=config.pad_token_id,
                )
                loss = loss / gradient_accumulation_steps

            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()

        t2 = time.perf_counter()
        if step % log_every == 0:
            wandb.log(
                {
                    "train/loss": gradient_accumulation_steps * loss.item(),
                    "train/grad_norm": grad_norm.item(),
                    "train/throughput": batch_size / (t2 - t1),
                    "train/lr": lr,
                },
                step=step,
            )

        step += 1

        if step % checkpoint_every == 0:
            checkpoint_path = os.path.join(run_dir, f"{kind}-{step:06d}.pt")
            print(f"Savings checkpoint to {checkpoint_path}")
            torch.save(
                {
                    "config": config.model_dump(),
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                checkpoint_path,
            )


if __name__ == "__main__":
    main()
