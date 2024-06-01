# wget -nc https://dl.fbaipublicfiles.com/textless_nlp/expresso/data/expresso.tar
# tar -xf expresso.tar

import pandas as pd
import tarfile
import torchaudio
from snac import SNAC
import torch
from torchaudio.transforms import Resample
from llmspeech.audio import flatten, unflatten
import os
from llmspeech.dataset import EXPRESSO_DATASET_DIR


device = "cuda"
codec = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
codec_sample_rate = 24000
expresso_sample_rate = 48000
resample = Resample(orig_freq=expresso_sample_rate, new_freq=codec_sample_rate).to(
    device
)
df = pd.read_csv("expresso/read_transcriptions.txt", sep="\t", header=None)
transcriptions = dict(zip(df[0], df[1]))

first = True

with torch.inference_mode():
    for name, transcript in transcriptions.items():
        # TODO(james) come back to this
        if "longform" in name:
            continue

        print(name, transcript)
        split = name.split("_")
        if len(split) == 3:
            speaker, style, num = split
        else:
            speaker, style, _, num = split

        audio_path = f"expresso/audio_48khz/read/{speaker}/{style}/base/{name}.wav"

        waveform, sample_rate = torchaudio.load(audio_path)
        assert sample_rate == expresso_sample_rate
        waveform = waveform.unsqueeze(0).to(device)

        waveform = resample(waveform)

        codes = codec.encode(waveform)

        flattened = flatten(codes).cpu()

        item = {
            "text": transcript,
            "audio_tokens": flattened,
            "style": style,
        }

        processed_path = os.path.join(EXPRESSO_DATASET_DIR, f"{name}.pt")
        torch.save(item, processed_path)

        if first:
            unflattened_codes = unflatten(flattened)
            unflattened_codes = list(map(lambda x: x.to(device), unflattened_codes))
            recons = codec.decode(unflattened_codes)
            torchaudio.save(
                f"preprocess-expresso-sanity-check.wav",
                recons.squeeze(0).cpu(),
                codec_sample_rate,
            )
            first = False
