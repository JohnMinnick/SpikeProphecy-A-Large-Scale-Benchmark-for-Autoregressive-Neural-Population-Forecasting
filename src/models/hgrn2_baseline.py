"""
HGRN2-inspired Teacher ANN baseline for spike-count forecasting.

Self-contained implementation (no flash-linear-attention dep). The FLA
library's package __init__ eagerly imports model adapters that collide
with the v22 container's transformers / triton versions; after two
integration attempts we vendor a clean implementation here.

This implementation is a simplified HGRN2: a **diagonal gated linear
RNN with state expansion**. Each hidden state dimension evolves
independently (diagonal recurrence -> biological-neuron-like inductive
prior). State expansion lifts the hidden dim by a configurable
`expand_ratio` so the model has capacity comparable to Mamba at matched
d_model. The forget gate is data-dependent and bounded in [0, 1].

Forward (per layer):
    x_t : (batch, d_model)
    i_t = W_i x_t              # input  -> (d_state,)
    f_t = sigmoid(W_f x_t)     # forget -> (d_state,), element-wise gate
    h_t = f_t * h_{t-1} + (1 - f_t) * i_t
    y_t = W_o h_t              # back to d_model

For T=10 bins the Python unroll is ~10 matmuls per layer per forward
pass, which is trivial compared to the projection layers.

Reference: "HGRN2: Gated Linear RNNs with State Expansion"
(Qin et al., arXiv:2404.07904) — this is a simplified re-implementation
matching the diagonal-recurrence + state-expansion structure, not a
bit-exact reproduction of their Triton-fused variant.
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class HGRN2Attention(nn.Module):
    """
    Diagonal gated linear RNN with state expansion.

    Args:
        hidden_size: Model dimension (d_model).
        num_heads: Reserved for API compat; currently unused
            (recurrence is scalar-per-dimension, so heads would just
            partition d_state). Default 1.
        expand_ratio: Multiplier for internal state dimension
            (d_state = hidden_size * expand_ratio). Default 2.
        use_bias: Whether projection layers have bias terms.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 1,
        expand_ratio: int = 2,
        use_bias: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expand_ratio = expand_ratio
        self.d_state = hidden_size * expand_ratio

        # Input projection: expand d_model -> d_state
        self.input_proj = nn.Linear(hidden_size, self.d_state, bias=use_bias)
        # Forget-gate projection
        self.forget_proj = nn.Linear(hidden_size, self.d_state, bias=use_bias)
        # Output projection: contract d_state -> d_model
        self.output_proj = nn.Linear(self.d_state, hidden_size, bias=use_bias)

        # Initialize forget gate to favor remembering early in training
        # (positive bias -> sigmoid > 0.5 -> keep more of h_{t-1}).
        # This is a standard trick for gated RNNs that helps gradient flow.
        if use_bias:
            nn.init.constant_(self.forget_proj.bias, 1.0)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> tuple:
        """
        Args:
            hidden_states: (batch, T, hidden_size).
            **kwargs: Consumed for API compat (e.g., attention_mask, past_kv).

        Returns:
            Tuple (output,) where output is (batch, T, hidden_size).
            Tuple shape matches FLA-style return; wrapper unpacks [0].
        """
        batch, T, _ = hidden_states.shape
        i = self.input_proj(hidden_states)      # (batch, T, d_state)
        f = torch.sigmoid(self.forget_proj(hidden_states))

        # Diagonal recurrence; sequential over T but T=10 is fine.
        h = torch.zeros(
            batch, self.d_state,
            device=hidden_states.device, dtype=hidden_states.dtype,
        )
        outs = []
        for t in range(T):
            h = f[:, t] * h + (1.0 - f[:, t]) * i[:, t]
            outs.append(h)
        out = torch.stack(outs, dim=1)          # (batch, T, d_state)
        out = self.output_proj(out)             # (batch, T, hidden_size)
        return (out,)


class HGRN2Block(nn.Module):
    """
    Single HGRN2 block with pre-norm residual connection.

    Mirrors MambaBlock's API for fair baseline comparison.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 1,
        expand_ratio: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.hgrn2 = HGRN2Attention(
            hidden_size=d_model,
            num_heads=num_heads,
            expand_ratio=expand_ratio,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        (x,) = self.hgrn2(hidden_states=x)
        x = self.dropout(x)
        return x + residual


class TeacherHGRN2(nn.Module):
    """
    HGRN2 teacher model for spike-count forecasting.

    Drop-in replacement for TeacherMamba with the same __init__ pattern
    and forward output shape. Minimal feature surface: no population
    coupling, no NegBin/ZIP heads, no GAC instrumentation — this model
    exists specifically as a diagonal-SSM comparison point against Mamba.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        num_heads: int = 1,
        expand_ratio: int = 2,
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

        self.hgrn2_blocks = nn.ModuleList([
            HGRN2Block(
                d_model=hidden_size,
                num_heads=num_heads,
                expand_ratio=expand_ratio,
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
            "TeacherHGRN2: input=%d, d_model=%d, layers=%d, "
            "heads=%d, expand_ratio=%d, d_state=%d, dropout=%.2f, "
            "output=%d, params=%d, layer_norm=%s, attention=%s, "
            "session_heads=%d",
            input_size, hidden_size, num_layers, num_heads, expand_ratio,
            hidden_size * expand_ratio, dropout, output_size,
            n_params, use_layer_norm, use_attention, n_sessions,
        )

    def forward(
        self, x: torch.Tensor, h0: Optional[torch.Tensor] = None,
        covariates: Optional[torch.Tensor] = None,
        session_id: Optional[str] = None,
    ) -> torch.Tensor:
        hidden = self.input_norm(self.input_proj(x))

        for block in self.hgrn2_blocks:
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
        """API compat with TeacherMamba — HGRN2 has no aux head."""
        return None

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], input_size: int,
        session_dims: Optional[Dict[str, int]] = None,
    ) -> "TeacherHGRN2":
        model_cfg = config.get("model", {})
        return cls(
            input_size=input_size,
            hidden_size=model_cfg.get("hidden_size", 256),
            num_layers=model_cfg.get("num_layers", 3),
            num_heads=model_cfg.get("num_heads", 1),
            expand_ratio=model_cfg.get("expand_ratio", 2),
            dropout=model_cfg.get("dropout", 0.2),
            output_size=model_cfg.get("output_size", None),
            use_layer_norm=model_cfg.get("use_layer_norm", False),
            use_attention=model_cfg.get("use_attention", False),
            session_dims=session_dims,
        )
