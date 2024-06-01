# llm.speech

Trying to join LLMs and speech models in a multi-task way. Inspired by GPT-4o and the details in the [blog post](https://laion.ai/notes/open-gpt-4-o/) from LAION.

## Progress

I think most of the following need to be ticked off on the route to a full multi-modal model:

- [x] Text-to-speech direction works in principle - although not very high quality at the moment and some issues that hopefully will get resolved with some iteration or scaling - see the `examples` directory for what the current model can (and can't) do.
- [ ] Warm starting text-to-speech training from a pretrained text LLM (will have to be a small one to fit on my 4090 dev box)
- [x] Text-to-speech finetuning on more expressive and higher quality datasets ([Expresso](https://github.com/facebookresearch/textlesslib/tree/main/examples/expresso/dataset)?)
- [ ] Speech-to-text direction
- [ ] Join text-to-speech and speech-to-text in a multi-task
- [ ] Interleaving like [SpiRit-LM](https://arxiv.org/abs/2402.05755)

There will probably be some more things to add along the way.

### 2024-05-31

- First training run on the MLS Eng dataset with a small model. 

https://github.com/jamesparsloe/llm.speech/assets/13669398/a9e1bcf0-ac81-4cbe-95c4-0f1906c6adbc

* Slightly cherry picked example.

### 2024-06-01

- Quick and dirty finetuning from the MLS Eng model on the Expresso dataset for 2000 steps

https://github.com/jamesparsloe/llm.speech/assets/13669398/fdae7771-864f-43c3-a5c3-8453e2d0100d

* Slightly cherry picked example.

## Getting Started

```sh
python3.11 -m venv env
source env/bin/activate
python -m pip install -e .[dev]
```

## Training

Currently just hardcoded for **text-to-speech**

### Datasets

```sh
python scripts/preprocess-mls-eng.py # downloads and processes the dataset MLS English dataset from HuggingFace
# wget -nc https://dl.fbaipublicfiles.com/textless_nlp/expresso/data/expresso.tar
# tar -xf expresso.tar
```

Train a small ~200M model on the MLS English dataset:

```sh
python -m llmspeech.train configs/small.yaml
```

## Demo

A simple demo for TT

```sh
python app.py
```

## TODO

- [x] Add kv-caching 😬
- [ ] Streaming inference with SNAC decoder
- [ ] Add DDP - just training on single 4090s currently
- [ ] `torch.compile` not completely working (CUDA Graphs aren't being used for some reason) I need to take a deeper look

## Acknowledgements

- [Andrej Karpathy](https://github.com/karpathy) for [nanoGPT](https://github.com/karpathy/nanoGPT) amongst all the other great things he's done
- [Hubert Siuzdak](https://github.com/hubertsiuzdak) for the awesome [SNAC](https://github.com/hubertsiuzdak/snac) codec
- [Julien Blanchon](https://github.com/julien-blanchon?tab=repositories) for creating the [snac_llm_parler_tts dataset](https://huggingface.co/datasets/blanchon/snac_llm_parler_tts) and saving me a lot of compute time (and effort)
