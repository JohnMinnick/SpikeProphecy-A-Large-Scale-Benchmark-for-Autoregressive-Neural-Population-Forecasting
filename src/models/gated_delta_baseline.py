"""
GatedDeltaNet-inspired Teacher ANN baseline — NON-diagonal SSM control.

Added 2026-04-22 per J. Eshraghian follow-up feedback: "Hgrn2 and mamba
are both p similar! If its possible to test Gated deltanet, thatd be a
useful result."

Motivation: HGRN2 and Mamba (both diagonal SSMs) converged to similar
performance on the benchmark, which is consistent with — but does not
prove — the diagonal-SSM inductive-prior hypothesis. A strong
non-diagonal modern SSM (GatedDeltaNet) run under the same recipe
gives us a control: if diagonal structure is what's doing the work,
GatedDeltaNet should underperform; if not, it should match.

Vendored implementation (no flash-linear-attention dep — FLA's
package init conflicts with the v22 container). Implements the
core gated delta rule with a *matrix* state (non-diagonal, per-head):

    q_t = W_q x_t, k_t = W_k x_t, v_t = W_v x_t   # per head
    g_t = sigmoid(W_g x_t)       # scalar forget gate per head
    beta_t = sigmoid(W_beta x_t) # scalar learning-rate gate per head
    k_t_norm = L2-normalize(k_t) # stability (common in delta nets)

    # predict current v from state, compute delta-error, gated update:
    predicted_v = S_{t-1} @ k_t_norm
    delta = v_t - predicted_v
    S_t = g_t * S_{t-1} + beta_t * outer(delta, k_t_norm)
    o_t = S_t @ q_t

State S_t is a (head_dim x head_dim) matrix per head → non-diagonal.

Reference: "Gated Delta Networks: Improving Mamba2 with Delta Rule"
(Yang et al., 2024). This is a simplified, chunked-Triton-free
re-implementation; sufficient for T=10 sequences.
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GatedDeltaAttention(nn.Module):
    """
    Gated delta rule layer (non-diagonal matrix state per head).

    Args:
        hidden_size: Model dimension (d_model).
        num_heads: Number of parallel heads (each with its own matrix state).
        head_dim: Per-head k/v dim; defaults to hidden_size // num_heads.
        use_bias: Whether projection layers have bias.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        head_dim: Optional[int] = None,
        use_bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        inner = self.num_heads * self.head_dim

        self.q_proj = nn.Linear(hidden_size, inner, bias=use_bias)
        self.k_proj = nn.Linear(hidden_size, inner, bias=use_bias)
        self.v_proj = nn.Linear(hidden_size, inner, bias=use_bias)
        # Scalar forget gate per head
        self.g_proj = nn.Linear(hidden_size, self.num_heads, bias=True)
        # Scalar learning-rate gate per head
        self.beta_proj = nn.Linear(hidden_size, self.num_heads, bias=True)
        # Output projection back to hidden_size
        self.o_proj = nn.Linear(inner, hidden_size, bias=use_bias)

        # Initialize gate biases: g toward "remember" (+1.0), beta toward
        # moderate update rate (0.0 → sigmoid 0.5). Standard gated-RNN init.
        nn.init.constant_(self.g_proj.bias, 1.0)
        nn.init.constant_(self.beta_proj.bias, 0.0)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> tuple:
        """
        Args:
            hidden_states: (batch, T, hidden_size).
            **kwargs: API compat, unused.

        Returns:
            (output,) where output is (batch, T, hidden_size).
        """
        B, T, _ = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, T, H, Dh)
        k = self.k_proj(hidden_states).view(B, T, H, Dh)
        v = self.v_proj(hidden_states).view(B, T, H, Dh)
        # Normalize keys (standard in delta nets for stability)
        k = F.normalize(k, dim=-1, eps=1e-5)

        g = torch.sigmoid(self.g_proj(hidden_states))        # (B, T, H)
        beta = torch.sigmoid(self.beta_proj(hidden_states))  # (B, T, H)

        # Matrix state per head: (B, H, Dh, Dh)
        S = torch.zeros(
            B, H, Dh, Dh,
            device=hidden_states.device, dtype=hidden_states.dtype,
        )
        outs = []
        for t in range(T):
            q_t = q[:, t]       # (B, H, Dh)
            k_t = k[:, t]       # (B, H, Dh)
            v_t = v[:, t]       # (B, H, Dh)
            g_t = g[:, t]       # (B, H)
            b_t = beta[:, t]    # (B, H)

            # Predict v from state: (B, H, Dh) = S @ k
            predicted_v = torch.einsum("bhij,bhj->bhi", S, k_t)
            delta = v_t - predicted_v                       # (B, H, Dh)
            # Outer product update: (B, H, Dh, Dh)
            update = delta.unsqueeze(-1) * k_t.unsqueeze(-2)
            # Gated state update
            g_exp = g_t[..., None, None]
            b_exp = b_t[..., None, None]
            S = g_exp * S + b_exp * update
            # Output: (B, H, Dh) = S @ q
            o_t = torch.einsum("bhij,bhj->bhi", S, q_t)
            outs.append(o_t)

        out = torch.stack(outs, dim=1)               # (B, T, H, Dh)
        out = out.reshape(B, T, H * Dh)              # (B, T, H*Dh)
        out = self.o_proj(out)                       # (B, T, hidden_size)
        return (out,)


class GatedDeltaBlock(nn.Module):
    """Single GatedDelta block with pre-norm residual."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        head_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = GatedDeltaAttention(
            hidden_size=d_model,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        (x,) = self.attn(hidden_states=x)
        x = self.dropout(x)
        return x + residual


class TeacherGatedDelta(nn.Module):
    """
    GatedDeltaNet teacher model for spike-count forecasting.

    Drop-in replacement for TeacherMamba / TeacherHGRN2 with the same
    __init__ pattern and forward output shape. Minimal feature surface:
    no population coupling, no NegBin/ZIP heads, no GAC instrumentation
    — this model exists specifically as a non-diagonal SSM control
    against Mamba + HGRN2.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        head_dim: Optional[int] = None,
        dropout: float = 0.2,
        output_size: Optional[int] = None,
        use_layer_norm: bool = False,
        use_attention: bool = False,
        session_dims: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.session_dims = session_dims

        if output_size is None:
            output_size = input_size
        self.output_size = output_size

        if session_dims is not None:
            self.session_output_projs = nn.ModuleDict()
            for sid, n_neurons in session_dims.items():
                self.session_output_projs[sid] = nn.Linear(
                    hidden_size, n_neurons,
                )
        else:
            self.session_output_projs = None

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.input_norm = (
            nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity()
        )
        self.output_norm = (
            nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity()
        )

        self.blocks = nn.ModuleList([
            GatedDeltaBlock(
                d_model=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_size)

        if use_attention:
            self.attn_query = nn.Linear(hidden_size, 1, bias=False)
        else:
            self.attn_query = None

        if session_dims is None:
            self.output_proj = nn.Linear(hidden_size, output_size)
        else:
            self.output_proj = None

        self.softplus = nn.Softplus()

        n_params = sum(p.numel() for p in self.parameters())
        n_sessions = len(session_dims) if session_dims else 0
        logger.info(
            "TeacherGatedDelta: input=%d, d_model=%d, layers=%d, "
            "num_heads=%d, head_dim=%d, dropout=%.2f, output=%d, "
            "params=%d, layer_norm=%s, attention=%s, session_heads=%d",
            input_size, hidden_size, num_layers, num_heads,
            head_dim if head_dim is not None else hidden_size // num_heads,
            dropout, output_size, n_params, use_layer_norm,
            use_attention, n_sessions,
        )

    def forward(
        self, x: torch.Tensor, h0: Optional[torch.Tensor] = None,
        covariates: Optional[torch.Tensor] = None,
        session_id: Optional[str] = None,
    ) -> torch.Tensor:
        hidden = self.input_norm(self.input_proj(x))

        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)

        if self.attn_query is not None:
            attn_scores = self.attn_query(hidden)
            attn_weights = torch.softmax(attn_scores, dim=1)
            context = (hidden * attn_weights).sum(dim=1)
        else:
            context = hidden[:, -1, :]

        context = self.output_norm(context)

        if self.session_output_projs is not None:
            assert session_id is not None, (
                "session_id is required when model uses session-specific heads"
            )
            raw_output = self.session_output_projs[session_id](context)
        else:
            raw_output = self.output_proj(context)

        return self.softplus(raw_output)

    def get_aux_output(self) -> Optional[torch.Tensor]:
        return None

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], input_size: int,
        session_dims: Optional[Dict[str, int]] = None,
    ) -> "TeacherGatedDelta":
        model_cfg = config.get("model", {})
        return cls(
            input_size=input_size,
            hidden_size=model_cfg.get("hidden_size", 256),
            num_layers=model_cfg.get("num_layers", 3),
            num_heads=model_cfg.get("num_heads", 4),
            head_dim=model_cfg.get("head_dim", None),
            dropout=model_cfg.get("dropout", 0.2),
            output_size=model_cfg.get("output_size", None),
            use_layer_norm=model_cfg.get("use_layer_norm", False),
            use_attention=model_cfg.get("use_attention", False),
            session_dims=session_dims,
        )
