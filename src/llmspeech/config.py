from pydantic import BaseModel


class Config(BaseModel):
    seed: int = 42

    batch_size: int = 64
    micro_batch_size: int = 8
    lr: float = 3e-4

    warmup_steps: int = 1000
    steps: int = 100_000
    checkpoint_every: int = 1000

    weight_decay: float = 0.1
    max_norm: float = 1.0
    betas: tuple[float, float] = (0.9, 0.95)

    max_grad_norm: float = 1.0

    n_text_tokens: int = 256

    n_quantizers: int = 3
    codebook_size: int = 4096

    kind: str = "gpt"

    d_model: int = 768
    n_heads: int = 12
    n_layer: int = 12
    bias: bool = False
    pad_vocab_size_multiple: int = 8
    dropout: float = 0.0
    use_rotary_emb: bool = True

    @property
    def grad_accumulation_steps(self):
        return self.batch_size // self.micro_batch_size

    @property
    def n_audio_tokens(self):
        return self.n_quantizers * self.codebook_size

    @property
    def n_tokens(self):
        return self.n_text_tokens + self.n_audio_tokens

    @property
    def bos_token_id(self):
        return self.n_tokens

    @property
    def boa_token_id(self):
        return self.n_tokens + 1

    @property
    def eos_token_id(self):
        return self.n_tokens + 1 + 1

    @property
    def pad_token_id(self):
        return self.n_tokens + 1 + 1 + 1

    @property
    def vocab_size(self):
        return self.n_tokens + 1 + 1 + 1 + 1
