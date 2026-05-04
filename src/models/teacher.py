"""
DEPRECATED — LSTM-based Teacher ANN for spike-count forecasting.

⚠️ This module is LEGACY. Mamba has replaced LSTM as the primary teacher
(ADR-0017). Use src/models/mamba_baseline.py for new work.
This file is kept for backward compatibility and re-exports shared
components from src/models/common.py.

Original architecture (soul.md §8.3):
    Input (batch, T, M)
    → Linear(M, hidden_size)    [input projection]
    → LayerNorm (optional)      [config: use_layer_norm]
    → LSTM(hidden_size, n_layers, dropout)
    → LayerNorm (optional)      [config: use_layer_norm]
    → Attention readout (optional) or last hidden state [config: use_attention]
    → Linear(hidden_size, M)    [output projection]
    → PopulationCouplingLayer (optional) [config: use_population_coupling]
    → Softplus                  [enforce λ > 0 for Poisson NLL]
    → Output (batch, M)         [predicted rates λ^ANN(t+1)]

Config-gated features (all backward-compatible, defaults match v1):
    - use_layer_norm: Add LayerNorm after input proj and before output (default False)
    - use_attention: Learned attention readout over all timesteps (default False)
    - use_population_coupling: Cross-neuron bottleneck MLP with residual connection
      (default False). Captures correlated ensemble firing patterns (ADR-0009 Batch C).
    - output_distribution: Output distribution type — "poisson" | "negbin" | "zip"
      (default "poisson"). Controls the output head and auxiliary parameters:
        - "poisson": Single rate output λ (original behavior)
        - "negbin": Rate λ + dispersion r (for overdispersed counts)
        - "zip": Rate λ + zero-inflation gate π (for excess zeros)
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Backward-compatible re-exports from common.py.
# These were originally defined here but are now shared across all teacher
# architectures (LSTM, LRU, Transformer, Mamba). Existing imports like
# `from src.models.teacher import PopulationCouplingLayer` continue to work.
from src.models.common import (  # noqa: F401
    VALID_DISTRIBUTIONS,
    PopulationCouplingLayer,
    create_teacher_model,
)


class TeacherLSTM(nn.Module):
    """
    LSTM teacher model for spike-count forecasting.

    Predicts non-negative rates λ(t+1) ∈ R^M_+ from a history window X_t.
    Optionally also predicts auxiliary parameters for alternative output
    distributions (dispersion for NegBin, zero-inflation gate for ZIP).

    Supports **session-specific read-in/read-out heads** (Phase 1 fix):
    when ``session_dims`` is provided, each session gets its own input
    and output projection layers.  The shared LSTM backbone operates in
    a common hidden space.

    Args:
        input_size: Number of input channels (M, or M_max in shared mode).
        hidden_size: LSTM hidden dimension.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout rate between LSTM layers (0 if num_layers=1).
        output_size: Number of output channels (defaults to input_size).
            Ignored when session_dims is provided (each session uses N_i).
        use_layer_norm: Apply LayerNorm after input projection and before
            output projection (default False).
        use_attention: Use learned attention readout over all LSTM timesteps
            instead of just the last hidden state (default False).
        output_distribution: Type of output distribution — "poisson" (default),
            "negbin" (Negative Binomial), or "zip" (Zero-Inflated Poisson).
        session_dims: Optional dict mapping session_id (str) → number of
            real neurons (int).  When provided, creates per-session
            input/output projections instead of shared ones.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
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
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.session_dims = session_dims  # Maps session_id → N_i

        # Validate output distribution
        if output_distribution not in VALID_DISTRIBUTIONS:
            raise ValueError(
                f"output_distribution must be one of {VALID_DISTRIBUTIONS}, "
                f"got '{output_distribution}'"
            )
        self.output_distribution = output_distribution

        # Default: predict same number of channels as input
        if output_size is None:
            output_size = input_size
        self.output_size = output_size

        # Covariate mode: "additive" (post-readout) or "temporal" (input concat)
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
            # Session-specific output heads (Phase 1 fix, ADR-0013):
            # Input proj stays SHARED because data loader pads all
            # sessions to M_max. Only output proj is per-session.
            self.session_input_projs = None  # Not used — shared input
            self.session_output_projs = nn.ModuleDict()
            for sid, n_neurons in session_dims.items():
                self.session_output_projs[sid] = nn.Linear(hidden_size, n_neurons)
            logger.info(
                "Session-specific output heads: %d sessions, dims=%s",
                len(session_dims),
                {k: v for k, v in sorted(session_dims.items())[:5]},
            )
        else:
            # Shared heads (original behavior)
            self.session_input_projs = None
            self.session_output_projs = None

        # Input projection: always shared (data is padded to M_max).
        if covariate_mode == "temporal" and n_covariates > 0:
            proj_input_size = input_size + n_covariates
        else:
            proj_input_size = input_size
        self.input_proj = nn.Linear(proj_input_size, hidden_size)

        # ------------------------------------------------------------------
        # LayerNorm (config-gated) — stabilizes gradients, lightweight regularization
        # ------------------------------------------------------------------
        self.input_norm = (
            nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity()
        )
        self.output_norm = (
            nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity()
        )

        # LSTM backbone
        # Dropout only applies between layers, so disable if single layer
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        # ------------------------------------------------------------------
        # Attention readout (config-gated) — learned query over all timesteps
        # ------------------------------------------------------------------
        if use_attention:
            # Single learnable query vector: projects each hidden state to a
            # scalar score, then softmax across time → weighted sum
            self.attn_query = nn.Linear(hidden_size, 1, bias=False)
        else:
            self.attn_query = None

        # ------------------------------------------------------------------
        # Output head — depends on output_distribution
        # (Only used in shared mode; session-specific uses session_output_projs)
        # ------------------------------------------------------------------
        if session_dims is None:
            self.output_proj = nn.Linear(hidden_size, output_size)
        else:
            self.output_proj = None  # Per-session output projs used instead

        # ------------------------------------------------------------------
        # Population coupling (config-gated, ADR-0009 Batch C)
        # Cross-neuron bottleneck MLP with residual, on raw logits
        # ------------------------------------------------------------------
        if use_population_coupling:
            self.coupling = PopulationCouplingLayer(
                num_channels=output_size,
                hidden_size=coupling_hidden_size,
            )
        else:
            self.coupling = None

        # Softplus enforces λ > 0 (required for all count distributions)
        self.softplus = nn.Softplus()

        # ------------------------------------------------------------------
        # Covariate projection (Option B additive fusion, ADR-0012)
        # Projects per-bin covariates into hidden space and adds to context
        # ------------------------------------------------------------------
        self.n_covariates = n_covariates
        if n_covariates > 0:
            self.covariate_proj = nn.Linear(n_covariates, hidden_size)
        else:
            self.covariate_proj = None

        # Auxiliary output head for NegBin (dispersion r) or ZIP (gate π)
        if output_distribution == "negbin":
            # Dispersion r > 0 via Softplus, initialized with positive bias
            # so r starts large (near-Poisson behavior)
            self.aux_proj = nn.Linear(hidden_size, output_size)
            # Initialize bias to ~2.0 so Softplus(2.0) ≈ 2.13, giving
            # moderate dispersion as starting point
            nn.init.constant_(self.aux_proj.bias, 2.0)
        elif output_distribution == "zip":
            # Gate π ∈ [0, 1] via Sigmoid, initialized with negative bias
            # so π starts near 0 (near-Poisson behavior)
            self.aux_proj = nn.Linear(hidden_size, output_size)
            # Initialize bias to -2.0 so Sigmoid(-2.0) ≈ 0.12 (low zero-inflation)
            nn.init.constant_(self.aux_proj.bias, -2.0)
        else:
            # Poisson: no auxiliary output needed
            self.aux_proj = None

        # Storage for auxiliary outputs (set during forward pass)
        # Allows trainer to access dispersion/gate without changing forward() signature
        self._aux_output: Optional[torch.Tensor] = None

        # Log parameter count
        n_params = sum(p.numel() for p in self.parameters())
        n_sessions = len(session_dims) if session_dims else 0
        logger.info(
            "TeacherLSTM: input=%d, hidden=%d, layers=%d, dropout=%.2f, "
            "output=%d, params=%d, layer_norm=%s, attention=%s, "
            "coupling=%s (h=%d), dist=%s, n_covariates=%d, cov_mode=%s, "
            "session_heads=%d",
            input_size, hidden_size, num_layers, dropout, output_size, n_params,
            use_layer_norm, use_attention, use_population_coupling,
            coupling_hidden_size, output_distribution, n_covariates,
            covariate_mode, n_sessions,
        )

    def forward(
        self, x: torch.Tensor, h0: Optional[torch.Tensor] = None,
        covariates: Optional[torch.Tensor] = None,
        session_id: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Forward pass: history window → predicted rates.

        Args:
            x: Input tensor of shape (batch, T, M) — history window
               of binned spike counts.  When session_dims is used,
               M = N_i (session's real neuron count, no padding).
            h0: Optional initial hidden state. If None, defaults to zeros.
            covariates: Optional covariate tensor.
                - additive mode: shape (batch, n_covariates) — target-bin only
                - temporal mode: shape (batch, T, n_covariates) — per-timestep
                When None, no covariate processing is applied.
            session_id: Required when session_dims is set. Selects the
                session-specific input/output projections.

        Returns:
            rates: Predicted non-negative rates, shape (batch, N_i) or
                   (batch, M).  For NegBin/ZIP, auxiliary parameters are
                   stored in self._aux_output via get_aux_output().
        """
        # Temporal covariate fusion: concatenate covariates to input at each
        # timestep before the input projection (ADR-0012)
        if (self.covariate_mode == "temporal" and covariates is not None
                and self.n_covariates > 0):
            # covariates: (batch, T, n_cov) -> concat with x -> (batch, T, M + n_cov)
            x = torch.cat([x, covariates], dim=-1)

        # ------------------------------------------------------------------
        # Input projection: always shared (data padded to M_max)
        # ------------------------------------------------------------------
        projected = self.input_norm(self.input_proj(x))

        # LSTM: (batch, T, hidden) → (batch, T, hidden)
        lstm_out, _ = self.lstm(projected, h0)

        # ------------------------------------------------------------------
        # Readout: attention over all timesteps, or last hidden state
        # ------------------------------------------------------------------
        if self.attn_query is not None:
            attn_scores = self.attn_query(lstm_out)
            attn_weights = torch.softmax(attn_scores, dim=1)
            context = (lstm_out * attn_weights).sum(dim=1)
        else:
            context = lstm_out[:, -1, :]

        # Normalize before output projection (Identity if LayerNorm disabled)
        context = self.output_norm(context)

        # Additive covariate fusion (Option B, ADR-0012)
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

        # Cross-neuron coupling on raw logits (before Softplus)
        # Only compatible with shared mode (fixed output_size)
        if self.coupling is not None and self.session_dims is None:
            raw_output = self.coupling(raw_output)

        # Softplus: enforce λ > 0 for count distribution compatibility
        rates = self.softplus(raw_output)

        # ------------------------------------------------------------------
        # Auxiliary output head (NegBin or ZIP)
        # Only for shared mode (aux_proj uses shared output_size)
        # ------------------------------------------------------------------
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
    ) -> "TeacherLSTM":
        """
        Create a TeacherLSTM from a config dictionary.

        Supports all new architecture config keys with backward-compatible defaults.

        Args:
            config: Config dict with 'model' section containing
                    hidden_size, num_layers, dropout, use_layer_norm,
                    use_attention, output_distribution, output_size.
            input_size: Number of input features (M + history features).
            session_dims: Optional dict mapping session_id → neuron count.

        Returns:
            Configured TeacherLSTM instance.
        """
        model_cfg = config.get("model", {})
        return cls(
            input_size=input_size,
            hidden_size=model_cfg.get("hidden_size", 128),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.2),
            output_size=model_cfg.get("output_size", None),
            use_layer_norm=model_cfg.get("use_layer_norm", False),
            use_attention=model_cfg.get("use_attention", False),
            use_population_coupling=model_cfg.get(
                "use_population_coupling", False
            ),
            coupling_hidden_size=model_cfg.get("coupling_hidden_size", 32),
            output_distribution=model_cfg.get("output_distribution", "poisson"),
            n_covariates=model_cfg.get("n_covariates", 0),
            covariate_mode=model_cfg.get("covariate_mode", "additive"),
            session_dims=session_dims,
        )


# NOTE: create_teacher_model is now in src.models.common and re-exported
# above via the backward-compatible import block. No definition here.
