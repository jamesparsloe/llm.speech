import torch
from torch import Tensor

# NOTE all hardcoded for SNAC 24kHz

CODEC_SAMPLE_RATE = 24000


def flatten(codes: list[Tensor]) -> Tensor:
    assert len(codes) == 3
    step = 7
    codes0 = codes[0].squeeze(0)
    codes1 = codes[1].squeeze(0)
    codes2 = codes[2].squeeze(0)

    flattened = torch.zeros(step * codes0.size(-1), dtype=torch.int32)
    flattened[0::step] = codes0
    flattened[1::step] = codes1[0::2]
    flattened[2::step] = codes2[0::4]
    flattened[3::step] = codes2[1::4]
    flattened[4::step] = codes1[1::2]
    flattened[5::step] = codes2[2::4]
    flattened[6::step] = codes2[3::4]

    return flattened


def unflatten(flattened):
    """
    Codes are flattened in this order:

        level 0    |       0       |
        level 1    |   1   |   4   |
        level 2    | 2 | 3 | 5 | 6 |

    I think this is a valid autoregressive flattening.
    """
    T = flattened.size(-1)
    step = 7
    assert T % step == 0
    codes0, codes1, codes2 = [], [], []

    for i in range(0, T, step):
        codes0.append(flattened[i])
        codes1.append(flattened[i + 1])
        codes2.append(flattened[i + 2])
        codes2.append(flattened[i + 3])
        codes1.append(flattened[i + 4])
        codes2.append(flattened[i + 5])
        codes2.append(flattened[i + 6])

    codes = list(map(lambda x: torch.tensor(x).unsqueeze(0), [codes0, codes1, codes2]))
    return codes


def validate_codes(codes: list[Tensor], codebook_size: int):
    for i, level in enumerate(codes):
        assert level.min() >= 0
        assert (
            level.max() < codebook_size
        ), f"Invalid token with value {level.max()} in level {i}"


def unflatten_and_remove_offsets(
    flattened: Tensor, *, n_text_tokens: int, codebook_size: int
):
    """Remove offsets we used when doing language modeling so that we can then feed into the SNAC decoder."""
    codes = unflatten(flattened)
    device = flattened.device

    offset = n_text_tokens

    for level in codes:
        level -= offset
        offset += codebook_size

    validate_codes(codes, codebook_size)

    codes = list(map(lambda x: x.to(device), codes))

    return codes
