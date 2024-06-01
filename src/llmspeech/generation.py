import torch
import torchaudio
from llmspeech.model import GPT
from llmspeech.text import tokenize, detokenize
from llmspeech.audio import unflatten_and_remove_offsets, CODEC_SAMPLE_RATE
from llmspeech.utils import count_parameters
import IPython.display as ipd
import time
from snac import SNAC
from torch import Tensor
import torch.nn.functional as F
from contextlib import nullcontext
from .model import GPT


def prefill(model: GPT, input_ids: Tensor, input_pos: Tensor, temperature=1.0, top_k=0):
    logits = model(input_ids=input_ids, input_pos=input_pos, num_last_tokens=1)
    logits = logits[:, -1]
    logits = logits / temperature

    if top_k > 1:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("inf")

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


def decode(model: GPT, cur_token: Tensor, input_pos: Tensor, temperature=1.0, top_k=1):
    logits = model(input_ids=cur_token, input_pos=input_pos, num_last_tokens=1)
    logits = logits[:, -1]

    # next_token = sample(logits, temperature=temperature, top_k=top_k)
    logits = logits / temperature

    if top_k > 1:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("inf")

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


@torch.inference_mode()
def generate(
    model: GPT,
    input_ids: Tensor,
    temperature: float = 1.0,
    top_k: int = 64,
    max_new_tokens: int = 20 * 84,
    compile: bool = True,
):
    global prefill, decode

    if compile:
        # TODO(james) still getting: skipping cudagraphs due to ['incompatible ops']
        prefill = torch.compile(prefill, fullgraph=True, dynamic=True)
        decode = torch.compile(decode, mode="reduce-overhead", fullgraph=True)

    B = input_ids.size(0)
    device = input_ids.device

    step = 0

    n_text_tokens = model.config.n_text_tokens
    n_quantizers = model.config.n_quantizers
    codebook_size = model.config.codebook_size

    period = 7
    rem_to_level = {0: 0, 1: 1, 2: 2, 3: 2, 4: 1, 5: 2, 6: 2}

    T = input_ids.size(-1)
    T_new = T + max_new_tokens

    empty = torch.empty(T_new, dtype=input_ids.dtype, device=device)
    empty[:T] = input_ids
    seq = empty

    batch_size = 1
    model.decoder.allocate_inference_cache(batch_size, device)

    input_pos = torch.arange(0, T, device=device)

    next_token = prefill(
        model, input_ids, input_pos, temperature=temperature, top_k=top_k
    )

    seq[T] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int32)
    new_tokens = []
    cur_token = next_token.view(1, -1)
    new_tokens.append(next_token.clone())

    while step < max_new_tokens - 1:
        with torch.nn.attention.sdpa_kernel(
            [torch.nn.attention.SDPBackend.MATH]
        ):  # Actually better for Inductor to codegen attention here
            next_token = decode(
                model, cur_token, input_pos, temperature=temperature, top_k=top_k
            )

        new_tokens.append(next_token.clone())
        cur_token = next_token.view(1, -1)
        input_pos += 1
        step += 1

    new_tokens = torch.cat(new_tokens, dim=-1)

    stop_idx = (new_tokens == model.config.eos_token_id).nonzero()

    if stop_idx.numel() > 0:
        stop_idx = stop_idx[0, -1].item()
        stop_reason = "eos"
    else:
        stop_idx = new_tokens.size(-1)
        stop_reason = "max_tokens"

    return new_tokens[:, :stop_idx]
