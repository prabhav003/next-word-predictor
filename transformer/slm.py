"""
slm.py
------

A small GPT-style decoder-only language model implemented in PyTorch.

Design goals:
- Simple enough to understand
- Fast enough to train on a single GPU
- Proper autoregressive next-token prediction
- Batched training
- Causal self-attention
- Weight tying
- KV-cache-ready generation
- Mixed-precision friendly

Suggested starting configuration:
    vocab_size = 16_000
    block_size = 256
    n_layer    = 6
    n_head     = 6
    n_embd     = 384

This is roughly a 20-30M parameter model depending on vocabulary size.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Configuration
# ============================================================

@dataclass
class SLMConfig:
    # Vocabulary
    vocab_size: int = 16_000

    # Maximum context length
    block_size: int = 256

    # Transformer
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

    # Feed-forward network multiplier
    ffn_mult: int = 4

    # Regularization
    dropout: float = 0.0

    # Special tokens
    bos_token_id: int = 0
    eos_token_id: int = 1

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                "n_embd must be divisible by n_head"
            )


# ============================================================
# LayerNorm
# ============================================================

class LayerNorm(nn.Module):
    """
    LayerNorm with an optional bias parameter.

    PyTorch's LayerNorm is already highly optimized, so we use it
    directly rather than manually implementing normalization.
    """

    def __init__(self, ndim: int):
        super().__init__()

        self.norm = nn.LayerNorm(
            ndim,
            elementwise_affine=True
        )

    def forward(self, x):
        return self.norm(x)


# ============================================================
# Causal Self Attention
# ============================================================

class CausalSelfAttention(nn.Module):

    def __init__(self, config: SLMConfig):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # One projection produces Q, K and V simultaneously.
        self.qkv = nn.Linear(
            config.n_embd,
            3 * config.n_embd,
            bias=True
        )

        self.out_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=True
        )

        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        """
        x:
            [B, T, C]

        past_kv:
            Optional cached K/V tensors.

        returns:
            output
            optionally (key, value) cache
        """

        B, T, C = x.shape

        # ----------------------------------------------------
        # QKV projection
        # ----------------------------------------------------

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # [B, T, C] -> [B, n_head, T, head_dim]

        head_dim = C // self.n_head

        q = q.view(
            B, T, self.n_head, head_dim
        ).transpose(1, 2)

        k = k.view(
            B, T, self.n_head, head_dim
        ).transpose(1, 2)

        v = v.view(
            B, T, self.n_head, head_dim
        ).transpose(1, 2)

        # ----------------------------------------------------
        # KV cache
        # ----------------------------------------------------

        if past_kv is not None:
            past_k, past_v = past_kv

            k = torch.cat(
                [past_k, k],
                dim=2
            )

            v = torch.cat(
                [past_v, v],
                dim=2
            )

        present = (k, v)

        # ----------------------------------------------------
        # Optimized scaled dot-product attention
        # ----------------------------------------------------

        # PyTorch automatically selects optimized CUDA
        # kernels when available.

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=(
                self.attn_dropout
                if self.training
                else 0.0
            ),
            is_causal=(
                past_kv is None
            ),
        )

        # [B, heads, T, head_dim]
        # -> [B, T, C]

        y = y.transpose(1, 2).contiguous()

        y = y.view(B, T, C)

        y = self.out_proj(y)

        y = self.resid_dropout(y)

        if use_cache:
            return y, present

        return y


# ============================================================
# Feed Forward Network
# ============================================================

class MLP(nn.Module):

    def __init__(self, config: SLMConfig):
        super().__init__()

        hidden_dim = config.ffn_mult * config.n_embd

        self.fc1 = nn.Linear(
            config.n_embd,
            hidden_dim
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            config.n_embd
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

    def forward(self, x):

        # GELU is commonly used in GPT-style models.

        x = self.fc1(x)

        x = F.gelu(
            x,
            approximate="tanh"
        )

        x = self.fc2(x)

        x = self.dropout(x)

        return x


# ============================================================
# Transformer Block
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(self, config: SLMConfig):
        super().__init__()

        self.ln1 = LayerNorm(
            config.n_embd
        )

        self.attn = CausalSelfAttention(
            config
        )

        self.ln2 = LayerNorm(
            config.n_embd
        )

        self.mlp = MLP(
            config
        )

    def forward(
        self,
        x,
        past_kv=None,
        use_cache=False
    ):

        # Pre-normalization architecture

        if use_cache:

            attn_out, present = self.attn(
                self.ln1(x),
                past_kv=past_kv,
                use_cache=True
            )

            x = x + attn_out

            x = x + self.mlp(
                self.ln2(x)
            )

            return x, present

        else:

            x = x + self.attn(
                self.ln1(x)
            )

            x = x + self.mlp(
                self.ln2(x)
            )

            return x


# ============================================================
# SLM
# ============================================================

class SLM(nn.Module):

    def __init__(self, config: SLMConfig):

        super().__init__()

        self.config = config

        # ----------------------------------------------------
        # Token embedding
        # ----------------------------------------------------

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd
        )

        # ----------------------------------------------------
        # Positional embedding
        # ----------------------------------------------------

        self.position_embedding = nn.Embedding(
            config.block_size,
            config.n_embd
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.n_layer)
            ]
        )

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        self.ln_f = LayerNorm(
            config.n_embd
        )

        # ----------------------------------------------------
        # Language-model head
        # ----------------------------------------------------

        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )

        # ----------------------------------------------------
        # Weight tying
        # ----------------------------------------------------

        # Input embeddings and output embeddings share weights.
        #
        # This saves parameters and is standard in many
        # language-model architectures.

        self.lm_head.weight = (
            self.token_embedding.weight
        )

        # ----------------------------------------------------
        # Initialization
        # ----------------------------------------------------

        self.apply(self._init_weights)

        # GPT-style scaled initialization for residual
        # projections.

        for name, param in self.named_parameters():

            if name.endswith("out_proj.weight"):
                nn.init.normal_(
                    param,
                    mean=0.0,
                    std=(
                        0.02
                        / (2 * config.n_layer) ** 0.5
                    )
                )

            if name.endswith("fc2.weight"):
                nn.init.normal_(
                    param,
                    mean=0.0,
                    std=(
                        0.02
                        / (2 * config.n_layer) ** 0.5
                    )
                )

    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------

    def _init_weights(
        self,
        module
    ):

        if isinstance(
            module,
            nn.Linear
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:

                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.Embedding
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    def num_parameters(
        self,
        non_embedding=True
    ):

        n_params = sum(
            p.numel()
            for p in self.parameters()
        )

        if non_embedding:

            n_params -= (
                self.position_embedding.weight.numel()
            )

        return n_params

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ):

        """
        input_ids:
            [B, T]

        targets:
            [B, T]

        Returns:

            logits:
                [B, T, vocab_size]

            loss:
                scalar, if targets are provided
        """

        B, T = input_ids.shape

        if T > self.config.block_size:

            raise ValueError(
                f"Sequence length {T} exceeds "
                f"block size {self.config.block_size}"
            )

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

        positions = torch.arange(
            0,
            T,
            device=input_ids.device
        )

        token_emb = self.token_embedding(
            input_ids
        )

        position_emb = self.position_embedding(
            positions
        )

        x = token_emb + position_emb

        x = self.dropout(x)

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        for block in self.blocks:

            x = block(x)

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        x = self.ln_f(x)

        logits = self.lm_head(x)

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1)
                ),
                targets.reshape(-1)
            )

        return logits, loss

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ):

        """
        Autoregressive generation.

        input_ids:
            [B, T]

        Returns:
            [B, T + generated_tokens]
        """

        self.eval()

        for _ in range(max_new_tokens):

            # Keep only the context the model can handle.

            input_context = input_ids[
                :, -self.config.block_size:
            ]

            logits, _ = self(
                input_context
            )

            # Only need the final timestep.

            logits = logits[:, -1, :]

            # Temperature

            if temperature <= 0:

                next_token = torch.argmax(
                    logits,
                    dim=-1,
                    keepdim=True
                )

            else:

                logits = logits / temperature

                # Top-k sampling

                if top_k is not None:

                    k = min(
                        top_k,
                        logits.size(-1)
                    )

                    values, _ = torch.topk(
                        logits,
                        k
                    )

                    min_value = values[:, -1].unsqueeze(-1)

                    logits = torch.where(
                        logits < min_value,
                        torch.full_like(
                            logits,
                            float("-inf")
                        ),
                        logits
                    )

                probabilities = F.softmax(
                    logits,
                    dim=-1
                )

                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1
                )

            input_ids = torch.cat(
                [
                    input_ids,
                    next_token
                ],
                dim=1
            )

            # Stop when EOS is generated.

            if eos_token_id is not None:

                if torch.all(
                    next_token.squeeze(-1)
                    == eos_token_id
                ):

                    break

        return input_ids


# ============================================================
# Optimizer
# ============================================================

def configure_optimizer(
    model: nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    betas=(0.9, 0.95),
):
    """
    Create AdamW with weight decay applied only to
    appropriate parameters.

    Biases and normalization parameters generally should
    not receive weight decay.
    """

    decay = []
    no_decay = []

    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        if (
            param.ndim >= 2
            and "embedding" not in name
        ):
            decay.append(param)

        else:
            no_decay.append(param)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": decay,
                "weight_decay": weight_decay,
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
            },
        ],
        lr=learning_rate,
        betas=betas,
        fused=torch.cuda.is_available(),
    )

    return optimizer


# ============================================================
# Batch creation
# ============================================================

def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str,
):
    """
    Sample random contiguous sequences from a tokenized corpus.

    data:
        1D tensor containing the entire tokenized dataset.

    Returns:

        x:
            [B, T]

        y:
            [B, T]

    where y is shifted by one token.
    """

    ix = torch.randint(
        0,
        len(data) - block_size - 1,
        (batch_size,)
    )

    x = torch.stack(
        [
            data[i:i + block_size]
            for i in ix
        ]
    )

    y = torch.stack(
        [
            data[i + 1:i + block_size + 1]
            for i in ix
        ]
    )

    return (
        x.to(
            device,
            non_blocking=True
        ),
        y.to(
            device,
            non_blocking=True
        ),
    )


# ============================================================
# Quick test
# ============================================================

if __name__ == "__main__":

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    config = SLMConfig(
        vocab_size=16_000,
        block_size=256,
        n_layer=6,
        n_head=6,
        n_embd=384,
    )

    model = SLM(config).to(device)

    print("=" * 60)
    print("Small Language Model")
    print("=" * 60)

    print(
        f"Parameters: "
        f"{model.num_parameters():,}"
    )

    print(
        f"Device: {device}"
    )

    # Fake batch for testing

    x = torch.randint(
        0,
        config.vocab_size,
        (8, 256),
        device=device
    )

    y = torch.randint(
        0,
        config.vocab_size,
        (8, 256),
        device=device
    )

    logits, loss = model(
        x,
        y
    )

    print(
        f"Input shape:  {x.shape}"
    )

    print(
        f"Logits shape: {logits.shape}"
    )

    print(
        f"Loss:         {loss.item():.4f}"
    )

    # Test generation

    prompt = torch.randint(
        0,
        config.vocab_size,
        (1, 10),
        device=device
    )

    generated = model.generate(
        prompt,
        max_new_tokens=20,
        temperature=0.8,
        top_k=50,
    )

    print(
        f"Generated shape: {generated.shape}"
    )

    print("=" * 60)