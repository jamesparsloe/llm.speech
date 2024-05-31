# llm.speech

Trying to join LLMs and speech models in a multi-task way. Inspired by GPT-4o and the details in the [blog post](https://laion.ai/notes/open-gpt-4-o/) from LAION.

## Progress

I think most of the following need to be ticked off on the route to a full multi-modal model:

- [x] Text-to-speech direction works in principle (although not very high quality at the moment and some issues that hopefully will get resolved with some iteration or scaling)
- [ ] Warm starting text-to-speech training from a pretrained text LLM (will have to be a small one to fit on my 4090 dev box)
- [ ] Text-to-speech finetuning on more expressive and higher quality datasets ([Expresso](https://github.com/facebookresearch/textlesslib/tree/main/examples/expresso/dataset)?)
- [ ] Speech-to-text direction
- [ ] Join text-to-speech and speech-to-text in a multi-task
- [ ] Interleaving like [SpiRit-LM](https://arxiv.org/abs/2402.05755)

There will probably be some more things to add along the way.

## Getting Started

```sh
python3.11 -m venv env
source env/bin/activate
python -m pip install -e .[dev]
```

## Training

Currently just hardcoded for **text-to-speech**

```sh
python -m llmspeech.preprocess # downloads and processes the dataset from HuggingFace
python -m llmspeech.train configs/small.yaml # train the small ~100M model
```

## TODO

- [ ] Add kv-caching 😬
- [ ] Streaming inference with SNAC decoder
- [ ] Add DDP - just training on single 4090s currently

## Acknowledgements

- [Andrej Karpathy](https://github.com/karpathy) for [nanoGPT](https://github.com/karpathy/nanoGPT) amongst all the other great things he's done
- [Hubert Siuzdak](https://github.com/hubertsiuzdak) for the awesome [SNAC](https://github.com/hubertsiuzdak/snac) codec
- [Julien Blanchon](https://github.com/julien-blanchon?tab=repositories) for creating the [snac_llm_parler_tts dataset](https://huggingface.co/datasets/blanchon/snac_llm_parler_tts) and saving me a lot of compute time (and effort)
