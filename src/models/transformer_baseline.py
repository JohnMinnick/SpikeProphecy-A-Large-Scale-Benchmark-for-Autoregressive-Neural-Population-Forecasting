"""
DEPRECATED — Transformer Teacher ANN baseline for spike-count forecasting.

⚠️ This module is LEGACY. Used once for architecture benchmarking (ADR-0016).
Mamba is the primary teacher (ADR-0017). Use src/models/mamba_baseline.py.

Original architecture:
    Input (batch, T, M)
    → Linear(M, d_model)         [input projection]
    → LayerNorm (optional)       [config: use_layer_norm]
    → Positional Encoding        [sinusoidal, length T]
    → TransformerEncoder (n_layers × EncoderLayer)
        - Causal multi-head self-attention (masked)
        - Feed-forward: d_model → d_ff → d_model
        - LayerNorm + residual connections
        - Dropout
    → Last timestep readout (or attention readout)
    → LayerNorm (optional)
    → Linear(d_model, M)         [output projection]
    → Softplus                   [enforce λ > 0 for Poisson NLL]
    → Output (batch, M)          [predicted rates λ^ANN(t+1)]

Drop-in replacement for TeacherLSTM / TeacherLRU — same __init__
signature pattern, same forward output shape, same from_config().

This provides a strong attention-based baseline to compare against
the LRU's linear recurrence approach. The causal mask ensures the
Transformer can only attend to past and present timesteps (no
future information leakage), matching the autoregressive task.
"""

import logging
import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.models.common import PopulationCouplingLayer, VALID_DISTRIBUTIONS

logger = logging.getLogger(__name__)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (Vaswani et al. 2017).

    Adds position-dependent signals to input embeddings so the
    Transformer can distinguish temporal ordering. Fixed (not learned)
    to avoid overfitting on short sequences.

    Args:
        d_model: Model embedding dimension.
        max_len: Maximum sequence length to pre-compute.
        dropout: Dropout rate applied after adding positional encoding.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute positional encoding matrix: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Geometric progression of wavelengths from 2π to max_len·2π
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer (not a parameter — no gradients)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (batch, T, d_model).

        Returns:
            Position-encoded tensor of shape (batch, T, d_model).
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TeacherTransformer(nn.Module):
    """
    Transformer teacher model for spike-count forecasting.

    Drop-in replacement for TeacherLSTM / TeacherLRU — same forward
    signature (batch, T, M) → (batch, M), same from_config() classmethod.

    Uses causal (autoregressive) attention masking to prevent future
    information leakage. The model attends to the full history window
    and uses the last timestep's representation for prediction.

    Args:
        input_size: Number of input channels (M, or M_max in shared mode).
        hidden_size: Transformer embedding dimension (d_model).
        num_layers: Number of Transformer encoder layers.
        n_heads: Number of attention heads. Must divide hidden_size.
        d_ff: Feed-forward inner dimension (default 4 × hidden_size).
        dropout: Dropout rate for attention and feed-forward layers.
        output_size: Number of output channels (defaults to input_size).
        use_layer_norm: Apply extra LayerNorm on input/output projections.
        use_attention: Use learned attention readout over all timesteps
            instead of last timestep (default False).
        use_population_coupling: Enable cross-neuron coupling MLP.
        coupling_hidden_size: Bottleneck dimension for coupling MLP.
        output_distribution: "poisson", "negbin", or "zip".
        n_covariates: Number of covariate features.
        covariate_mode: "additive" or "temporal".
        session_dims: Optional dict mapping session_id → neuron count
            for per-session output heads.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        n_heads: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.2,
        output_size: Optional[int] = None,
        use_layer_norm: bool = False,
        use_attention: bool = False,
        use_population_coupling: bool = False,
        coupling_hidden_size: int = 32,
        output_distribution: str = "poisson",
        n_covariates: int = 0,
        covariate_mode: str = "additive",
        session_dims: Optional[Dict[str, int]] = None,
        bidirectional: bool = False,
        masked_bin_frac: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.use_attention = use_attention
        self.session_dims = session_dims
        # NDT2-style options:
        # bidirectional: drop the causal attention mask (full
        #   bidirectional attention across the H-bin window).
        # masked_bin_frac: random fraction of input bins to zero out
        #   during training (BERT-style masked-LM objective at the
        #   forward pass; loss is unchanged — model still predicts the
        #   next-bin target). 0.0 disables.
        self.bidirectional = bool(bidirectional)
        self.masked_bin_frac = float(masked_bin_frac)

        # Default feed-forward dimension: 4× model dim (standard Transformer)
        if d_ff is None:
            d_ff = 4 * hidden_size
        self.d_ff = d_ff

        # Validate output distribution
        if output_distribution not in VALID_DISTRIBUTIONS:
            raise ValueError(
                f"output_distribution must be one of {VALID_DISTRIBUTIONS}, "
                f"got '{output_distribution}'"
            )
        self.output_distribution = output_distribution

        # Validate n_heads divides hidden_size
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"n_heads ({n_heads})"
            )

        # Default: predict same number of channels as input
        if output_size is None:
            output_size = input_size
        self.output_size = output_size

        # Covariate mode validation
        if covariate_mode not in ("additive", "temporal"):
            raise ValueError(
                f"covariate_mode must be 'additive' or 'temporal', "
                f"got '{covariate_mode}'"
            )
        self.covariate_mode = covariate_mode

        # ------------------------------------------------------------------
        # Input/output projections: session-specific or shared
        # ------------------------------------------------------------------
        if session_dims is not None:
            # Session-specific output heads (same pattern as LRU/LSTM)
            self.session_input_projs = None
            self.session_output_projs = nn.ModuleDict()
            for sid, n_neurons in session_dims.items():
                self.session_output_projs[sid] = nn.Linear(
                    hidden_size, n_neurons
                )
            logger.info(
                "Session-specific output heads: %d sessions, dims=%s",
                len(session_dims),
                {k: v for k, v in sorted(session_dims.items())[:5]},
            )
        else:
            self.session_input_projs = None
            self.session_output_projs = None

        # Input projection: always shared (data padded to M_max)
        # Temporal mode concatenates covariates → wider input
        if covariate_mode == "temporal" and n_covariates > 0:
            proj_input_size = input_size + n_covariates
        else:
            proj_input_size = input_size
        self.input_proj = nn.Linear(proj_input_size, hidden_size)

        # ------------------------------------------------------------------
        # LayerNorm (config-gated)
        # ------------------------------------------------------------------
        self.input_norm = (
            nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity()
        )
        self.output_norm = (
            nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity()
        )

        # ------------------------------------------------------------------
        # Positional encoding
        # ------------------------------------------------------------------
        self.pos_encoder = SinusoidalPositionalEncoding(
            d_model=hidden_size, max_len=512, dropout=dropout,
        )

        # ------------------------------------------------------------------
        # Transformer encoder with causal masking
        # ------------------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,       # (batch, T, d_model) format
            norm_first=True,        # Pre-norm (more stable training)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,  # Required for causal mask compat
        )

        # ------------------------------------------------------------------
        # Attention readout (config-gated)
        # ------------------------------------------------------------------
        if use_attention:
            self.attn_query = nn.Linear(hidden_size, 1, bias=False)
        else:
            self.attn_query = None

        # ------------------------------------------------------------------
        # Output head
        # ------------------------------------------------------------------
        if session_dims is None:
            self.output_proj = nn.Linear(hidden_size, output_size)
        else:
            self.output_proj = None

        # ------------------------------------------------------------------
        # Population coupling (config-gated)
        # ------------------------------------------------------------------
        if use_population_coupling:
            self.coupling = PopulationCouplingLayer(
                num_channels=output_size,
                hidden_size=coupling_hidden_size,
            )
        else:
            self.coupling = None

        # Softplus enforces λ > 0
        self.softplus = nn.Softplus()

        # ------------------------------------------------------------------
        # Covariate projection (additive mode)
        # ------------------------------------------------------------------
        self.n_covariates = n_covariates
        if n_covariates > 0:
            self.covariate_proj = nn.Linear(n_covariates, hidden_size)
        else:
            self.covariate_proj = None

        # Auxiliary output head for NegBin or ZIP
        if output_distribution == "negbin":
            self.aux_proj = nn.Linear(hidden_size, output_size)
            nn.init.constant_(self.aux_proj.bias, 2.0)
        elif output_distribution == "zip":
            self.aux_proj = nn.Linear(hidden_size, output_size)
            nn.init.constant_(self.aux_proj.bias, -2.0)
        else:
            self.aux_proj = None

        # Storage for auxiliary outputs
        self._aux_output: Optional[torch.Tensor] = None

        # Log parameter count
        n_params = sum(p.numel() for p in self.parameters())
        n_sessions = len(session_dims) if session_dims else 0
        logger.info(
            "TeacherTransformer: input=%d, d_model=%d, layers=%d, "
            "heads=%d, d_ff=%d, dropout=%.2f, output=%d, params=%d, "
            "layer_norm=%s, attention=%s, coupling=%s (h=%d), "
            "dist=%s, n_covariates=%d, cov_mode=%s, session_heads=%d",
            input_size, hidden_size, num_layers, n_heads, d_ff,
            dropout, output_size, n_params, use_layer_norm, use_attention,
            use_population_coupling, coupling_hidden_size,
            output_distribution, n_covariates, covariate_mode, n_sessions,
        )

    def _generate_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Generate a causal attention mask for autoregressive forecasting.

        The mask prevents each position from attending to future positions.
        Position i can only attend to positions 0..i.

        Args:
            T: Sequence length.
            device: Target device for the mask tensor.

        Returns:
            Float mask of shape (T, T) with -inf for blocked positions.
        """
        # Upper triangular mask: True where attention is BLOCKED
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1)
        # Convert to additive mask: 0 for allowed, -inf for blocked
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(
        self, x: torch.Tensor, h0: Optional[torch.Tensor] = None,
        covariates: Optional[torch.Tensor] = None,
        session_id: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Forward pass: history window → predicted rates.

        Args:
            x: Input tensor of shape (batch, T, M).
            h0: Unused, kept for API compatibility.
            covariates: Optional covariate tensor.
                - additive mode: shape (batch, n_covariates)
                - temporal mode: shape (batch, T, n_covariates)
            session_id: Required when session_dims is set.

        Returns:
            rates: Predicted non-negative rates, shape (batch, M) or
                   (batch, N_i) for session-specific heads.
        """
        # Temporal covariate fusion: concat covariates at each timestep
        if (self.covariate_mode == "temporal" and covariates is not None
                and self.n_covariates > 0):
            x = torch.cat([x, covariates], dim=-1)

        # ------------------------------------------------------------------
        # NDT2-style masked-bin training: zero out a random fraction of
        # input bins so the model must reconstruct from context.
        # Only applied during training (self.training).
        # ------------------------------------------------------------------
        if self.training and self.masked_bin_frac > 0.0:
            B, T_x, _ = x.shape
            keep_prob = 1.0 - self.masked_bin_frac
            bin_keep = (torch.rand(B, T_x, 1, device=x.device) < keep_prob).float()
            x = x * bin_keep

        # ------------------------------------------------------------------
        # Input projection + positional encoding
        # ------------------------------------------------------------------
        projected = self.input_norm(self.input_proj(x))  # (batch, T, d_model)
        projected = self.pos_encoder(projected)

        # ------------------------------------------------------------------
        # Transformer encoder. If bidirectional=True, drop the causal mask
        # so the encoder attends across the full H-bin window in both
        # directions (NDT2 / BERT-style). Otherwise apply the standard
        # autoregressive causal mask.
        # ------------------------------------------------------------------
        T = projected.size(1)
        if self.bidirectional:
            encoded = self.transformer_encoder(projected)
        else:
            causal_mask = self._generate_causal_mask(T, projected.device)
            encoded = self.transformer_encoder(
                projected, mask=causal_mask, is_causal=True,
            )

        # ------------------------------------------------------------------
        # Readout: attention over all timesteps, or last timestep
        # ------------------------------------------------------------------
        if self.attn_query is not None:
            attn_scores = self.attn_query(encoded)      # (batch, T, 1)
            attn_weights = torch.softmax(attn_scores, dim=1)
            context = (encoded * attn_weights).sum(dim=1)  # (batch, d_model)
        else:
            context = encoded[:, -1, :]  # Last timestep

        # Normalize before output projection
        context = self.output_norm(context)

        # Additive covariate fusion
        if (self.covariate_mode == "additive" and covariates is not None
                and self.covariate_proj is not None):
            context = context + self.covariate_proj(covariates)

        # ------------------------------------------------------------------
        # Output projection: session-specific or shared
        # ------------------------------------------------------------------
        if self.session_output_projs is not None:
            assert session_id is not None, (
                "session_id is required when model uses session-specific heads"
            )
            output_proj = self.session_output_projs[session_id]
            raw_output = output_proj(context)
        else:
            raw_output = self.output_proj(context)

        # Cross-neuron coupling (shared mode only)
        if self.coupling is not None and self.session_dims is None:
            raw_output = self.coupling(raw_output)

        # Softplus: enforce λ > 0
        rates = self.softplus(raw_output)

        # Auxiliary output head (NegBin or ZIP)
        if self.aux_proj is not None and self.session_dims is None:
            raw_aux = self.aux_proj(context)
            if self.output_distribution == "negbin":
                self._aux_output = self.softplus(raw_aux)
            elif self.output_distribution == "zip":
                self._aux_output = torch.sigmoid(raw_aux)
        else:
            self._aux_output = None

        return rates

    def get_aux_output(self) -> Optional[torch.Tensor]:
        """
        Get the auxiliary output from the last forward pass.

        Returns:
            For "negbin": dispersion parameter r, shape (batch, M).
            For "zip": zero-inflation gate π, shape (batch, M).
            For "poisson": None.
        """
        return self._aux_output

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], input_size: int,
        session_dims: Optional[Dict[str, int]] = None,
    ) -> "TeacherTransformer":
        """
        Create a TeacherTransformer from a config dictionary.

        Mirrors TeacherLSTM.from_config() / TeacherLRU.from_config().

        Args:
            config: Config dict with 'model' section containing
                    hidden_size, num_layers, n_heads, d_ff, dropout,
                    use_layer_norm, use_attention, output_distribution.
            input_size: Number of input features (M + history features).
            session_dims: Optional dict mapping session_id → neuron count.

        Returns:
            Configured TeacherTransformer instance.
        """
        model_cfg = config.get("model", {})
        return cls(
            input_size=input_size,
            hidden_size=model_cfg.get("hidden_size", 256),
            num_layers=model_cfg.get("num_layers", 3),
            n_heads=model_cfg.get("n_heads", 8),
            d_ff=model_cfg.get("d_ff", None),
            dropout=model_cfg.get("dropout", 0.2),
            output_size=model_cfg.get("output_size", None),
            use_layer_norm=model_cfg.get("use_layer_norm", False),
            use_attention=model_cfg.get("use_attention", False),
            use_population_coupling=model_cfg.get(
                "use_population_coupling", False,
            ),
            coupling_hidden_size=model_cfg.get("coupling_hidden_size", 32),
            output_distribution=model_cfg.get(
                "output_distribution", "poisson",
            ),
            n_covariates=model_cfg.get("n_covariates", 0),
            covariate_mode=model_cfg.get("covariate_mode", "additive"),
            session_dims=session_dims,
            bidirectional=model_cfg.get("bidirectional", False),
            masked_bin_frac=model_cfg.get("masked_bin_frac", 0.0),
        )
