import math
import random

import numpy as np
import torch


def seed_all(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


def warmup_then_cosine_decay(
    step: int, *, warmup_steps: int, steps: int, min_lr: float, max_lr: float
):
    if step < warmup_steps:
        return min_lr + step * (max_lr - min_lr) / (warmup_steps)
    elif step > steps:
        return min_lr
    else:
        decay_ratio = (step - warmup_steps) / (steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)


def decompile_state_dict(state_dict):
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
