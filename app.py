import gradio as gr
from llmspeech.model import GPT
from llmspeech import generation
from llmspeech.text import tokenize, detokenize
from snac import SNAC
import torch
from llmspeech.audio import unflatten_and_remove_offsets, CODEC_SAMPLE_RATE
import numpy as np
import torchaudio
import time
import os

device = "cuda"

codec = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)

MODELS = {
    "small-mls": "gpt-small-055000.pt",
    "small-expresso-finetune": "gpt-002000.pt",
}

MODEL_CACHE = {}

prev_model_name = None

INT16_MAX = 32767


@torch.inference_mode()
def generate(model_name: str, text: str, temperature: float, top_k: int):
    global prev_model_name

    model = MODEL_CACHE.get(model_name)

    if prev_model_name is not None and prev_model_name != model_name:
        MODEL_CACHE[prev_model_name].cpu()  # offload to CPU

    if model is None:
        huggingface_name = MODELS[model_name]
        model = GPT.from_huggingface(huggingface_name).eval().to(device)
        MODEL_CACHE[model_name] = model

    config = model.config

    prev_model_name = model_name

    input_ids = [config.bos_token_id] + tokenize(text) + [config.boa_token_id]
    input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        output_ids = generation.generate(
            model, input_ids, temperature=temperature, top_k=top_k
        )

    # HACK if decoding failed somehow and we didn't generate a multiple of 7
    step = 7
    T_out = output_ids.size(-1)
    rem = T_out % step
    if rem > 0:
        output_ids = output_ids[:, :-rem]

    codes = unflatten_and_remove_offsets(
        output_ids[0],
        n_text_tokens=config.n_text_tokens,
        codebook_size=config.codebook_size,
    )

    waveform = codec.decode(codes)
    waveform = waveform.squeeze(0).cpu()

    path = f"/tmp/{temperature=}-{top_k=}-{int(time.time())}.wav"

    torchaudio.save(path, waveform, CODEC_SAMPLE_RATE)
    return path


with gr.Blocks(
    title="llm.speech",
) as demo:
    with gr.Row():
        gr.Markdown("""# llm.speech""")
    with gr.Row():
        with gr.Column():
            model_name = gr.Dropdown(
                list(MODELS.keys()),
                label="Model",
                info="Select a model to use.",
                value=list(MODELS.keys())[0],
                interactive=True,
            )
            text = gr.Textbox(
                label="Text",
                info="",
                interactive=True,
            )
            temperature = gr.Slider(
                minimum=0.1, maximum=2.0, value=1.0, step=0.01, label="Temperature"
            )
            top_k = gr.Slider(minimum=0, maximum=4096, value=64, step=1, label="Top-k")
            generate_button = gr.Button(value="Generate")

        with gr.Column():
            audio_output = gr.Audio(
                label="Generated", type="filepath", interactive=False
            )

        generate_button.click(
            inputs=[model_name, text, temperature, top_k],
            outputs=[audio_output],
            fn=generate,
        )


demo.launch()
