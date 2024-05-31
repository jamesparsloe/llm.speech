import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import hf_hub_download
from rotary_embedding_torch import RotaryEmbedding
from torch import Tensor

from .config import Config


class MHA(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        *,
        bias: bool = False,
        dropout: float = 0.0,
        causal: bool = False,
        use_rotary_emb: bool = False,
        rotary_dim: int = 32,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.causal = causal
        self.use_rotary_emb = use_rotary_emb

        self.Wqkv = nn.Linear(dim, 3 * dim, bias=bias)

        self.out_proj = nn.Linear(dim, dim, bias=bias)

        if self.use_rotary_emb:
            self.rotary_emb = RotaryEmbedding(dim=rotary_dim)

    def forward(
        self,
        x: Tensor,
        x_kv: Tensor | None = None,
        need_weights: bool = False,
        offset: int = 0,
        attn_mask: Tensor | None = None,
    ):
        B, T, d = x.size()
        dtype = x.dtype

        qkv = self.Wqkv(x)
        qkv = rearrange(
            qkv, "B T (three h d) -> B three h T d", three=3, h=self.n_heads
        )
        q, k, v = qkv.unbind(dim=1)

        if self.use_rotary_emb:
            q = self.rotary_emb.rotate_queries_or_keys(q, offset=offset)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        attn_weights = None

        if need_weights:
            attn_weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            attn_weights = F.softmax(attn_weights, dim=-1)
            out = attn_weights @ v
        else:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal,
            )

        out = self.out_proj(rearrange(out, "B h T d -> B T (h d)"))

        return out, attn_weights


class MLP(nn.Module):
    def __init__(self, *, d_model: int, bias: bool, dropout: float, act=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model, bias=bias)
        self.act = act()
        self.fc2 = nn.Linear(4 * d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool,
        dropout: float,
        use_rotary_emb: bool,
    ):
        super().__init__()

        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = MHA(
            d_model,
            n_heads,
            bias=bias,
            dropout=dropout,
            use_rotary_emb=use_rotary_emb,
            causal=True,
        )

        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model=d_model, bias=bias, dropout=dropout)

    def forward(self, x: Tensor, x_kv: Tensor):
        attn_out, attn_weights = self.attn(self.attn_norm(x))
        x = x + attn_out

        x = x + self.mlp(self.mlp_norm(x))

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        bias: bool,
        dropout: float,
        use_rotary_emb: bool = True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model,
                    n_heads,
                    bias=bias,
                    dropout=dropout,
                    use_rotary_emb=use_rotary_emb,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, x_kv: Tensor | None = None):
        for block in self.blocks:
            x = block(x, x_kv=x_kv)
        x = self.norm(x)
        return x


class GPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.config = config

        d_model = config.d_model
        n_layer = config.n_layer
        vocab_size = config.vocab_size

        pad_vocab_size_multiple = config.pad_vocab_size_multiple

        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (
                vocab_size % pad_vocab_size_multiple
            )

        self.emb = nn.Embedding(vocab_size, d_model)

        self.decoder = Decoder(
            n_layers=n_layer,
            d_model=d_model,
            n_heads=config.n_heads,
            bias=config.bias,
            dropout=config.dropout,
            use_rotary_emb=config.use_rotary_emb,
        )

        self.lm_head = nn.Linear(d_model, vocab_size, bias=config.bias)

        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @staticmethod
    def from_huggingface(filename: str):
        path = hf_hub_download(repo_id="jamesparsloe/llm.speech", filename=filename)

        return GPT.from_pretrained(path)

    @staticmethod
    def from_pretrained(path: str):
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint["model"]
        config = Config(**checkpoint["config"])
        model = GPT(config)
        _ = model.load_state_dict(state_dict, strict=False)
        return model

    def forward(
        self,
        *,
        input_ids: Tensor,
        num_last_tokens: int = 0,
    ):
        B, T = input_ids.size()
        device = input_ids.device

        emb = self.emb(input_ids)

        hidden_states = self.decoder(emb)

        # decoding
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]

        logits = self.lm_head(hidden_states)

        return logits

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        max_new_tokens: int = 700,
        do_masking: bool = False,
    ):
        B = input_ids.size(0)
        device = input_ids.device

        step = 0

        n_text_tokens = self.config.n_text_tokens
        n_quantizers = self.config.n_quantizers
        codebook_size = self.config.codebook_size

        period = 7
        rem_to_level = {0: 0, 1: 1, 2: 2, 3: 2, 4: 1, 5: 2, 6: 2}

        prefix_len = input_ids.size(-1)

        while step < max_new_tokens:
            logits = self(input_ids=input_ids, num_last_tokens=1)
            logits = logits[:, -1]

            if do_masking:
                rem = step % period

                level = rem_to_level[rem]

                # FIXME come back to this - I'm not convinced it works properly
                # interestngly sometimes we just keep decoding in text "space" since they
                # share the same Embedding and output layer and all that

                ids = torch.arange(logits.size(-1), device=device)
                mask = torch.where(
                    (ids >= n_text_tokens + level * codebook_size)
                    & (ids < n_text_tokens + (level + 1) * codebook_size),
                    0.0,
                    -float("inf"),
                )

                logits = logits + mask

            # next_token = sample(logits, temperature=temperature, top_k=top_k)
            logits = logits / temperature

            if top_k > 1:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)

            if torch.any(next_token >= self.config.n_tokens):
                break

            # next_token = repeat(next_token, "1 -> B 1", B=B)

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            step += 1

        # slice prefix
        return input_ids[:, prefix_len:]


def build_optimizer(module: nn.Module, *, weight_decay: float, lr: float, betas):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in module.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, fused=True)

    return optimizer
