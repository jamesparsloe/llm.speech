import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from llmspeech.audio import flatten, unflatten, unflatten_and_remove_offsets
from llmspeech.text import detokenize, tokenize

DATASET_DIR = os.path.expanduser("~/.cache/datasets/snac_llm_parler_tts")
os.makedirs(DATASET_DIR, exist_ok=True)


class SNACDataset(Dataset):
    def __init__(
        self,
        root: str,
        *,
        n_text_tokens: int,
        codebook_size: int,
        bos_token_id: int,
        boa_token_id: int,
        eos_token_id: int,
    ):
        self.root = root
        self.paths = os.listdir(root)

        self.n_text_tokens = n_text_tokens
        self.codebook_size = codebook_size
        self.bos_token_id = bos_token_id
        self.boa_token_id = boa_token_id
        self.eos_token_id = eos_token_id

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.paths[idx])
        item = torch.load(path)
        text = item["text"]

        input_ids = tokenize(text)
        input_ids = [self.bos_token_id] + input_ids + [self.boa_token_id]

        input_ids = torch.tensor(input_ids)

        audio_tokens = item["audio_tokens"]

        unflattened = unflatten(audio_tokens)
        offset = self.n_text_tokens
        unflattened_with_offsets = []
        for level in unflattened:
            level = level + offset
            unflattened_with_offsets.append(level)
            offset += self.codebook_size

        audio_tokens = flatten(unflattened_with_offsets)
        audio_tokens = F.pad(audio_tokens, (0, 1), value=self.eos_token_id)

        token_ids = torch.cat([input_ids, audio_tokens], dim=-1)

        return token_ids


if __name__ == "__main__":
    import torchaudio
    from snac import SNAC

    from .config import Config

    device = "cuda:0"

    codec = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
    codec_sample_rate = 24000

    config = Config()

    dataset = SNACDataset(
        DATASET_DIR,
        n_text_tokens=config.n_text_tokens,
        codebook_size=config.codebook_size,
        bos_token_id=config.bos_token_id,
        boa_token_id=config.boa_token_id,
        eos_token_id=config.eos_token_id,
    )

    print(f"{len(dataset)} items in dataset")

    token_ids = dataset[0].to(device)

    boa_idx = (token_ids == config.boa_token_id).nonzero().item()
    text_ids = token_ids[1:boa_idx]  # strip BOS
    audio_ids = token_ids[boa_idx + 1 : -1]  # strip EOS

    codes = unflatten_and_remove_offsets(
        audio_ids,
        n_text_tokens=config.n_text_tokens,
        codebook_size=config.codebook_size,
    )

    text = detokenize(text_ids)

    with torch.inference_mode():
        waveform = codec.decode(codes)

    print(text)

    torchaudio.save(
        "dataset-sanity-check.wav", waveform.squeeze(0).cpu(), codec_sample_rate
    )
