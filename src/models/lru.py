"""
DEPRECATED — LRU (Linear Recurrent Unit) Teacher ANN for spike-count forecasting.

⚠️ This module is LEGACY. Mamba has replaced LRU v2 as the primary teacher
(ADR-0017). Use src/models/mamba_baseline.py for new work.

Original architecture (ADR-0011):
    Input (batch, T, M)
    → Linear(M, hidden_size)    [input projection]
    → LayerNorm (optional)      [config: use_layer_norm]
    → LRU stack (num_layers × LRUCell)
        - h_t = Λ * h_{t-1} + B * u_t  (complex diagonal recurrence)
        - Λ = exp(-exp(ν) + i·exp(θ))  (guaranteed stable: |λ| < 1)
        - Gated mode: ν_t, θ_t from Linear(u_t)
    → Attention readout (optional) or last hidden state
    → LayerNorm (optional)
    → Linear(hidden_size, M)    [output projection]
    → PopulationCouplingLayer (optional)
    → Softplus                  [enforce λ > 0 for Poisson NLL]
    → Output (batch, M)         [predicted rates λ^ANN(t+1)]

Drop-in replacement for TeacherLSTM — same __init__ signature, same forward
output shape, same from_config() classmethod.
"""

import logging
import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.models.common import PopulationCouplingLayer, VALID_DISTRIBUTIONS

logger = logging.getLogger(__name__)


class LRUCell(nn.Module):
    """
    Single LRU layer with complex diagonal state.

    Implements the recurrence:
        h_t = Λ * h_{t-1} + B * u_t

    where Λ is a complex diagonal matrix parameterized for guaranteed
    stability (|λ_i| < 1 for all i).

    Stable parameterization:
        Λ = exp(-exp(ν) + i·exp(θ))
        |Λ| = exp(-exp(ν)) < 1  since exp(ν) > 0

    Args:
        input_size: Dimension of input features at each timestep.
        hidden_size: Dimension of complex hidden state (number of
            diagonal entries in Λ).
        gated: If True, ν and θ are content-dependent (computed from
            input at each timestep via learned linear projections).
            If False, ν and θ are global learnable parameters.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        gated: bool = False,
        r_min: float = 0.8,
        r_max: float = 0.99,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gated = gated

        # B projection: maps real input to complex hidden space
        # We project to 2*hidden_size and split into real/imag parts
        # Scaled by 1/sqrt(hidden_size) for stable signal propagation
        self.B_proj = nn.Linear(input_size, 2 * hidden_size)
        with torch.no_grad():
            self.B_proj.weight.mul_(1.0 / math.sqrt(hidden_size))

        if gated:
            # Content-dependent eigenvalues: input → (ν_t, θ_t)
            self.gate_nu = nn.Linear(input_size, hidden_size)
            self.gate_theta = nn.Linear(input_size, hidden_size)
            # Bias gate_nu so initial |λ| falls in [r_min, r_max]
            # |λ| = exp(-exp(ν)), so ν = ln(-ln(r))
            with torch.no_grad():
                u = torch.rand(hidden_size) * (r_max - r_min) + r_min
                nu_bias = torch.log(-torch.log(u))
                self.gate_nu.bias.copy_(nu_bias)
                # Bias gate_theta for uniform phase in [0, 2π]
                self.gate_theta.bias.copy_(
                    torch.rand(hidden_size) * 2 * math.pi
                )
        else:
            # Ring initialization (Orvieto et al. 2023):
            #   |λ| = exp(-exp(ν)) should be uniform in [r_min, r_max]
            #   θ should be uniform in [0, 2π]
            # Invert |λ| = exp(-exp(ν)) → ν = ln(-ln(|λ|))
            u = torch.rand(hidden_size) * (r_max - r_min) + r_min
            nu_init = torch.log(-torch.log(u))
            theta_init = torch.rand(hidden_size) * 2 * math.pi
            self.nu = nn.Parameter(nu_init)
            self.theta = nn.Parameter(theta_init)

        # Output projection: map complex hidden state back to real
        # Takes real part of hidden state (hidden_size) → output
        self.C_proj = nn.Linear(hidden_size, input_size)

    @torch.amp.autocast('cuda', enabled=False)
    def _compute_lambda(
        self, x_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute complex eigenvalues Λ with guaranteed |λ| < 1.

        Note: Autocast disabled because complex ops (1j * exp(θ))
        don't support BFloat16 on some GPUs (e.g. RTX 3090).

        Args:
            x_t: Input at current timestep, shape (batch, input_size).
                 Only used in gated mode.

        Returns:
            Complex tensor of shape (hidden_size,) or (batch, hidden_size).
        """
        if self.gated:
            # Content-dependent: ν_t, θ_t from input (cast to float32)
            assert x_t is not None, "Gated mode requires input x_t"
            nu = self.gate_nu(x_t.float())       # (batch, hidden_size)
            theta = self.gate_theta(x_t.float())  # (batch, hidden_size)
        else:
            # Global parameters (already float32)
            nu = self.nu     # (hidden_size,)
            theta = self.theta  # (hidden_size,)

        # Stable parameterization: |λ| = exp(-exp(ν)) < 1
        lambda_ = torch.exp(-torch.exp(nu) + 1j * torch.exp(theta))
        return lambda_

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sequential scan over time dimension.

        Note: Autocast disabled because torch.complex() and complex
        arithmetic don't support BFloat16 on some GPUs (e.g. RTX 3090).
        All operations run in float32; output is cast back to the
        caller's dtype to preserve gradient flow.

        Args:
            x: Input tensor of shape (batch, T, input_size).

        Returns:
            Output tensor of shape (batch, T, input_size) — real-valued
            projections of all hidden states across time.
        """
        # Remember original dtype so we can cast output back
        orig_dtype = x.dtype
        x = x.float()  # Force float32 for complex ops

        batch, T, _ = x.shape

        # Project input to complex space: (batch, T, 2*hidden) → split
        B_out = self.B_proj(x)  # (batch, T, 2*hidden_size)
        B_real = B_out[..., :self.hidden_size]
        B_imag = B_out[..., self.hidden_size:]
        u = torch.complex(B_real, B_imag)  # (batch, T, hidden_size)

        # Initialize hidden state as complex zeros
        h = torch.zeros(
            batch, self.hidden_size,
            dtype=torch.cfloat, device=x.device,
        )

        # Sequential scan — collect all hidden states
        outputs = []
        for t in range(T):
            x_t = x[:, t, :]  # (batch, input_size)
            lambda_ = self._compute_lambda(x_t if self.gated else None)
            h = lambda_ * h + u[:, t, :]
            # Take real part for output projection
            outputs.append(self.C_proj(h.real))  # (batch, input_size)

        # Stack along time: (batch, T, input_size)
        result = torch.stack(outputs, dim=1)
        # Cast back to original dtype for compatibility with AMP
        return result.to(orig_dtype)


class TeacherLRU(nn.Module):
    """
    LRU teacher model for spike-count forecasting.

    Drop-in replacement for TeacherLSTM — same __init__ params, same
    forward signature (batch, T, M) → (batch, M), same from_config().

    Predicts non-negative rates λ(t+1) ∈ R^M_+ from a history window X_t.
    Optionally also predicts auxiliary parameters for alternative output
    distributions (dispersion for NegBin, zero-inflation gate for ZIP).

    Supports **session-specific read-in/read-out heads** (Phase 1 fix):
    when ``session_dims`` is provided, each session gets its own input
    and output projection layers, so neurons at the same index in
    different sessions are NOT treated as the same cell.  The shared
    LRU backbone operates in a common hidden space.

    Args:
        input_size: Number of input channels (M, or M_max in shared mode).
        hidden_size: LRU hidden dimension.
        num_layers: Number of stacked LRU layers.
        dropout: Dropout rate between LRU layers (0 if num_layers=1).
        output_size: Number of output channels (defaults to input_size).
            Ignored when session_dims is provided (each session uses N_i).
        use_layer_norm: Apply LayerNorm after input projection and before
            output projection (default False).
        use_attention: Use learned attention readout over all timesteps
            instead of just the last hidden state (default False).
        use_population_coupling: Enable cross-neuron coupling MLP
            (default False).
        coupling_hidden_size: Bottleneck dimension for coupling MLP
            (default 32).
        output_distribution: Type of output distribution — "poisson"
            (default), "negbin", or "zip".
        gated: Whether to use content-dependent eigenvalues in
            LRU cells (default False).
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
        gated: bool = False,
        n_covariates: int = 0,
        covariate_mode: str = "additive",
        session_dims: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.gated = gated
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
            # sessions to M_max. Only output proj is per-session:
            # Linear(hidden, N_i) predicts only real neurons.
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
        # Temporal mode concatenates covariates at each timestep → wider input.
        if covariate_mode == "temporal" and n_covariates > 0:
            proj_input_size = input_size + n_covariates
        else:
            proj_input_size = input_size
        self.input_proj = nn.Linear(proj_input_size, hidden_size)

        # ------------------------------------------------------------------
        # LayerNorm (config-gated) — stabilizes gradients
        # ------------------------------------------------------------------
        self.input_norm = (
            nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity()
        )
        self.output_norm = (
            nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity()
        )

        # ------------------------------------------------------------------
        # LRU backbone — stack of LRUCell layers with dropout, LayerNorm,
        # and residual connections (stabilizes deep stacks at scale)
        # ------------------------------------------------------------------
        self.lru_layers = nn.ModuleList()
        self.lru_dropouts = nn.ModuleList()
        self.lru_norms = nn.ModuleList()  # Per-layer norm (prevents drift)
        lru_dropout = dropout if num_layers > 1 else 0.0
        for i in range(num_layers):
            self.lru_layers.append(
                LRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    gated=gated,
                )
            )
            # LayerNorm per LRU layer (stabilizes activations between layers)
            self.lru_norms.append(nn.LayerNorm(hidden_size))
            # Dropout between layers (not after the last one)
            if i < num_layers - 1:
                self.lru_dropouts.append(nn.Dropout(lru_dropout))
            else:
                self.lru_dropouts.append(nn.Identity())

        # ------------------------------------------------------------------
        # Attention readout (config-gated)
        # ------------------------------------------------------------------
        if use_attention:
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
        # Projects per-bin covariates into hidden space and adds to context.
        # Mirrors TeacherLSTM's covariate_proj exactly.
        # ------------------------------------------------------------------
        self.n_covariates = n_covariates
        if n_covariates > 0:
            self.covariate_proj = nn.Linear(n_covariates, hidden_size)
        else:
            self.covariate_proj = None

        # Auxiliary output head for NegBin or ZIP
        # NOTE: aux_proj uses output_size (shared). Session-specific aux
        # would need per-session aux heads — deferred unless needed.
        if output_distribution == "negbin":
            self.aux_proj = nn.Linear(hidden_size, output_size)
            nn.init.constant_(self.aux_proj.bias, 2.0)
        elif output_distribution == "zip":
            self.aux_proj = nn.Linear(hidden_size, output_size)
            nn.init.constant_(self.aux_proj.bias, -2.0)
        else:
            self.aux_proj = None

        # Storage for auxiliary outputs (set during forward pass)
        self._aux_output: Optional[torch.Tensor] = None

        # Log parameter count
        n_params = sum(p.numel() for p in self.parameters())
        n_sessions = len(session_dims) if session_dims else 0
        logger.info(
            "TeacherLRU: input=%d, hidden=%d, layers=%d, dropout=%.2f, "
            "output=%d, params=%d, layer_norm=%s, attention=%s, "
            "coupling=%s (h=%d), dist=%s, gated=%s, n_covariates=%d, "
            "cov_mode=%s, session_heads=%d",
            input_size, hidden_size, num_layers, dropout, output_size,
            n_params, use_layer_norm, use_attention,
            use_population_coupling, coupling_hidden_size,
            output_distribution, gated, n_covariates, covariate_mode,
            n_sessions,
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
            h0: Unused, kept for API compatibility with TeacherLSTM.
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

        # LRU stack with residual connections and per-layer norm:
        # (batch, T, hidden) → (batch, T, hidden)
        lru_out = projected
        for lru_layer, lru_drop, lru_norm in zip(
            self.lru_layers, self.lru_dropouts, self.lru_norms,
        ):
            # Residual connection: stabilizes deep stacks (3+ layers)
            lru_out = lru_out + lru_drop(lru_norm(lru_layer(lru_out)))

        # ------------------------------------------------------------------
        # Readout: attention over all timesteps, or last hidden state
        # ------------------------------------------------------------------
        if self.attn_query is not None:
            attn_scores = self.attn_query(lru_out)  # (batch, T, 1)
            attn_weights = torch.softmax(attn_scores, dim=1)
            context = (lru_out * attn_weights).sum(dim=1)  # (batch, hidden)
        else:
            context = lru_out[:, -1, :]  # Last hidden state

        # Normalize before output projection
        context = self.output_norm(context)

        # Additive covariate fusion (Option B, ADR-0012)
        # Only applied in additive mode — temporal mode already concat'd to input
        if (self.covariate_mode == "additive" and covariates is not None
                and self.covariate_proj is not None):
            context = context + self.covariate_proj(covariates)

        # ------------------------------------------------------------------
        # Output projection: session-specific or shared
        # ------------------------------------------------------------------
        if self.session_output_projs is not None:
            # Session-specific mode: select the right projection
            assert session_id is not None, (
                "session_id is required when model uses session-specific heads"
            )
            output_proj = self.session_output_projs[session_id]
            raw_output = output_proj(context)
        else:
            # Shared mode: single output projection
            raw_output = self.output_proj(context)

        # Cross-neuron coupling on raw logits (before Softplus)
        # NOTE: coupling is shared and uses output_size, so it's only
        # compatible with shared mode. Skipped in session-specific mode.
        if self.coupling is not None and self.session_dims is None:
            raw_output = self.coupling(raw_output)

        # Softplus: enforce λ > 0
        rates = self.softplus(raw_output)

        # ------------------------------------------------------------------
        # Auxiliary output head (NegBin or ZIP)
        # NOTE: aux_proj uses shared output_size — only for shared mode.
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
    ) -> "TeacherLRU":
        """
        Create a TeacherLRU from a config dictionary.

        Supports all architecture config keys with backward-compatible
        defaults. Mirrors TeacherLSTM.from_config() exactly.

        Args:
            config: Config dict with 'model' section containing
                    hidden_size, num_layers, dropout, use_layer_norm,
                    use_attention, output_distribution, output_size, gated.
            input_size: Number of input features (M + history features).
            session_dims: Optional dict mapping session_id → neuron count.
                When provided, creates per-session input/output projections.

        Returns:
            Configured TeacherLRU instance.
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
                "use_population_coupling", False,
            ),
            coupling_hidden_size=model_cfg.get("coupling_hidden_size", 32),
            output_distribution=model_cfg.get(
                "output_distribution", "poisson",
            ),
            gated=model_cfg.get("gated", False),
            n_covariates=model_cfg.get("n_covariates", 0),
            covariate_mode=model_cfg.get("covariate_mode", "additive"),
            session_dims=session_dims,
        )
