"""
Mamba-based Teacher ANN baseline for spike-count forecasting.

Architecture:
    Input (batch, T, M)
    → Linear(M, d_model)         [input projection]
    → LayerNorm (optional)       [config: use_layer_norm]
    → Mamba blocks (n_layers × Mamba layer)
        - Selective state space with input-dependent gating
        - d_model → d_state × d_conv → d_model
    → Last timestep readout (or attention readout)
    → LayerNorm (optional)
    → Linear(d_model, M)         [output projection]
    → Softplus                   [enforce λ > 0 for Poisson NLL]
    → Output (batch, M)          [predicted rates λ^ANN(t+1)]

Drop-in replacement for TeacherLSTM / TeacherLRU / TeacherTransformer
— same __init__ signature pattern, same forward output shape.

NOTE: Requires the `mamba-ssm` package which only works on Linux + CUDA.
This model is designed for NRP deployment only (won't run on Windows).

Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with
Selective State Spaces" (2023).
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.models.common import PopulationCouplingLayer, VALID_DISTRIBUTIONS

logger = logging.getLogger(__name__)

# Lazy import of mamba_ssm (only available on Linux + CUDA)
_MAMBA_AVAILABLE = False
Mamba = None  # Will be set by import below
try:
    from mamba_ssm import Mamba
    _MAMBA_AVAILABLE = True
    logger.info("mamba-ssm imported successfully (Mamba v1/v2 API)")
except ImportError:
    try:
        # mamba-ssm v2.2+ may expose Mamba2 instead of Mamba
        from mamba_ssm import Mamba2 as Mamba  # noqa: F811
        _MAMBA_AVAILABLE = True
        logger.info("mamba-ssm imported successfully (Mamba2 API)")
    except ImportError:
        try:
            # Some versions nest under modules
            from mamba_ssm.modules.mamba_simple import Mamba  # noqa: F811
            _MAMBA_AVAILABLE = True
            logger.info("mamba-ssm imported successfully (mamba_simple API)")
        except ImportError as e:
            logger.info(
                "mamba-ssm not installed or import failed: %s. "
                "MambaBlock requires Linux + CUDA.", e
            )


class MambaBlock(nn.Module):
    """
    Single Mamba block with residual connection and LayerNorm.

    Wraps the core Mamba layer with pre-norm residual connection,
    following the standard deep SSM block pattern.

    Args:
        d_model: Model dimension.
        d_state: SSM state dimension (default 16).
        d_conv: Local convolution width (default 4).
        expand: Expansion factor for inner dimension (default 2).
        dropout: Dropout rate after Mamba layer.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not _MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm package is required for MambaBlock. "
                "Install with: pip install mamba-ssm. "
                "NOTE: Only works on Linux + CUDA."
            )

        # Pre-norm residual: LayerNorm → Mamba → residual + dropout
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Mamba block with residual connection.

        Args:
            x: Input tensor of shape (batch, T, d_model).

        Returns:
            Output tensor of shape (batch, T, d_model).
        """
        # Pre-norm residual connection
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x + residual


class InstrumentedMambaBlock(nn.Module):
    """
    Mamba block wrapper that captures internal SSM signals (Δ, B, C).

    Uses PyTorch forward hooks on the Mamba layer's internal linear
    projections to capture the pre-SSM signals without modifying the
    compiled CUDA kernel.  The captured signals are stored for use
    in mechanism alignment losses (GAC-SNN, Tier 2F).

    The Mamba layer internally has:
    - in_proj: projects input to [z, x] (gate and SSM input)
    - x_proj: projects SSM input to [dt, B, C] (the three dynamic params)
    - dt_proj: projects dt_rank to d_inner (the discretization step)

    We capture the output of x_proj and dt_proj to get the pre-SSM
    signals for alignment.

    Args:
        mamba_block: An existing MambaBlock to wrap.
    """

    def __init__(self, mamba_block: MambaBlock):
        super().__init__()
        self.block = mamba_block

        # Storage for captured signals
        self._bc_signal = None  # [dt_rank, B, C] from x_proj output

        # Register hooks on internal projections
        self._hooks = []
        mamba_layer = self.block.mamba

        # CRITICAL: Disable mamba-ssm's fused CUDA kernel (mamba_inner_fn).
        # The fused path calls x_proj/dt_proj via raw F.linear(), bypassing
        # nn.Module.forward() and our hooks. The slow path calls them as
        # proper module forward() calls, which triggers our hooks.
        if hasattr(mamba_layer, 'use_fast_path'):
            mamba_layer.use_fast_path = False
            logger.info(
                "Disabled Mamba fast path for signal capture "
                "(slow path enables forward hooks)"
            )

        # Hook on x_proj to capture [dt_rank, B, C] combined.
        # NOTE: We extract ALL three signals (delta, B, C) from x_proj.
        # dt_proj is NOT hooked because mamba-ssm's slow path uses direct
        # weight access (self.dt_proj.weight @ dt.t()) instead of calling
        # the module's forward(), so hooks never fire on dt_proj.
        if hasattr(mamba_layer, 'x_proj'):
            h = mamba_layer.x_proj.register_forward_hook(
                self._capture_bc_hook
            )
            self._hooks.append(h)

    def _capture_bc_hook(self, module, input, output):
        """Forward hook to capture [dt_rank, B, C] from x_proj.

        All three signals (delta, B, C) are extracted from x_proj:
        - dt_rank portion → Δ (pre-dt_proj, pre-softplus)
        - d_state portion → B (input projection)
        - d_state portion → C (output projection)
        """
        self._bc_signal = output.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (delegates to wrapped block)."""
        return self.block(x)

    def get_signals(self) -> dict:
        """
        Get captured SSM signals from the last forward pass.

        Returns:
            Dict with key 'bc' (may be None if hook didn't fire).
        """
        return {"bc": self._bc_signal}

    def remove_hooks(self):
        """Remove all forward hooks (for cleanup)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class TeacherMamba(nn.Module):
    """
    Mamba teacher model for spike-count forecasting.

    Drop-in replacement for TeacherLSTM / TeacherLRU / TeacherTransformer.
    Uses Mamba's selective state space mechanism with input-dependent gating
    for efficient long-sequence modeling.

    Args:
        input_size: Number of input channels (M, or M_max in shared mode).
        hidden_size: Model embedding dimension (d_model).
        num_layers: Number of stacked Mamba blocks.
        d_state: SSM state dimension (default 16).
        d_conv: Local convolution width (default 4).
        expand: Expansion factor for inner dimension (default 2).
        dropout: Dropout rate.
        output_size: Number of output channels (defaults to input_size).
        use_layer_norm: Apply extra LayerNorm on input/output projections.
        use_attention: Use learned attention readout over all timesteps
            instead of last timestep (default False).
        use_population_coupling: Enable cross-neuron coupling MLP.
        coupling_hidden_size: Bottleneck dimension for coupling MLP.
        output_distribution: "poisson", "negbin", or "zip".
        n_covariates: Number of covariate features.
        covariate_mode: "additive" or "temporal".
        session_dims: Optional dict mapping session_id → neuron count.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
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
        self.session_dims = session_dims

        # Validate output distribution
        if output_distribution not in VALID_DISTRIBUTIONS:
            raise ValueError(
                f"output_distribution must be one of {VALID_DISTRIBUTIONS}, "
                f"got '{output_distribution}'"
            )
        self.output_distribution = output_distribution

        # Default output size
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
            self.session_input_projs = None
            self.session_output_projs = nn.ModuleDict()
            for sid, n_neurons in session_dims.items():
                self.session_output_projs[sid] = nn.Linear(
                    hidden_size, n_neurons,
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
        # Mamba blocks
        # ------------------------------------------------------------------
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(
                d_model=hidden_size,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        # Final norm after all blocks
        self.final_norm = nn.LayerNorm(hidden_size)

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
            "TeacherMamba: input=%d, d_model=%d, layers=%d, "
            "d_state=%d, d_conv=%d, expand=%d, dropout=%.2f, "
            "output=%d, params=%d, layer_norm=%s, attention=%s, "
            "coupling=%s (h=%d), dist=%s, n_covariates=%d, "
            "cov_mode=%s, session_heads=%d",
            input_size, hidden_size, num_layers, d_state, d_conv,
            expand, dropout, output_size, n_params, use_layer_norm,
            use_attention, use_population_coupling, coupling_hidden_size,
            output_distribution, n_covariates, covariate_mode, n_sessions,
        )

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
        # Temporal covariate fusion
        if (self.covariate_mode == "temporal" and covariates is not None
                and self.n_covariates > 0):
            x = torch.cat([x, covariates], dim=-1)

        # ------------------------------------------------------------------
        # Input projection
        # ------------------------------------------------------------------
        projected = self.input_norm(self.input_proj(x))  # (batch, T, d_model)

        # ------------------------------------------------------------------
        # Mamba blocks (no causal mask needed — Mamba is inherently causal)
        # ------------------------------------------------------------------
        hidden = projected
        for block in self.mamba_blocks:
            hidden = block(hidden)
        # Final layer norm
        hidden = self.final_norm(hidden)

        # ------------------------------------------------------------------
        # Readout: attention over all timesteps, or last timestep
        # ------------------------------------------------------------------
        if self.attn_query is not None:
            attn_scores = self.attn_query(hidden)
            attn_weights = torch.softmax(attn_scores, dim=1)
            context = (hidden * attn_weights).sum(dim=1)
        else:
            context = hidden[:, -1, :]  # Last timestep

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

    def enable_instrumentation(self) -> None:
        """
        Enable SSM signal instrumentation for mechanism alignment.

        Wraps each MambaBlock with InstrumentedMambaBlock to capture
        Δ, B, C pre-SSM projections via forward hooks.  Call this
        ONCE before the first distillation forward pass.
        """
        instrumented = nn.ModuleList()
        for block in self.mamba_blocks:
            if isinstance(block, InstrumentedMambaBlock):
                instrumented.append(block)  # Already wrapped
            else:
                instrumented.append(InstrumentedMambaBlock(block))
        self.mamba_blocks = instrumented
        logger.info(
            "Mamba instrumentation enabled: %d blocks wrapped",
            len(instrumented),
        )

    def disable_instrumentation(self) -> None:
        """Remove instrumentation hooks and unwrap blocks."""
        unwrapped = nn.ModuleList()
        for block in self.mamba_blocks:
            if isinstance(block, InstrumentedMambaBlock):
                block.remove_hooks()
                unwrapped.append(block.block)
            else:
                unwrapped.append(block)
        self.mamba_blocks = unwrapped
        logger.info("Mamba instrumentation disabled")

    def get_internal_signals(self) -> Dict[str, Optional[torch.Tensor]]:
        """
        Get captured SSM signals from the last forward pass.

        Returns signals from the LAST Mamba block (deepest layer),
        which has the most abstract representation.

        Returns:
            Dict with keys:
            - 'delta': Δ signal (batch × T, d_inner) or None
            - 'B': B signal portion of x_proj output, or None
            - 'C': C signal portion of x_proj output, or None
        """
        result = {"delta": None, "B": None, "C": None}

        # Find the last instrumented block
        last_block = None
        for block in reversed(list(self.mamba_blocks)):
            if isinstance(block, InstrumentedMambaBlock):
                last_block = block
                break

        if last_block is None:
            return result

        signals = last_block.get_signals()

        # Split x_proj output into [dt_rank, B, C]
        # All three signals come from x_proj; dt_proj hook is not used
        # because mamba-ssm calls dt_proj via direct weight access.
        bc = signals.get("bc")
        if bc is not None:
            mamba_layer = last_block.block.mamba
            dt_rank = getattr(mamba_layer, 'dt_rank', 0)
            d_state = getattr(mamba_layer, 'd_state', 16)

            # x_proj output is [dt_rank + d_state + d_state]
            if bc.shape[-1] >= dt_rank + 2 * d_state:
                # Δ signal: dt_rank portion (pre-dt_proj, pre-softplus).
                # The learned SignalProjector absorbs the dt_proj linear
                # transformation and the softplus nonlinearity.
                result["delta"] = bc[..., :dt_rank]
                result["B"] = bc[..., dt_rank:dt_rank + d_state]
                result["C"] = bc[..., dt_rank + d_state:]

        return result

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], input_size: int,
        session_dims: Optional[Dict[str, int]] = None,
    ) -> "TeacherMamba":
        """
        Create a TeacherMamba from a config dictionary.

        Mirrors TeacherLSTM.from_config() / TeacherLRU.from_config().

        Args:
            config: Config dict with 'model' section containing
                    hidden_size, num_layers, d_state, d_conv, expand,
                    dropout, use_layer_norm, output_distribution.
            input_size: Number of input features (M + history features).
            session_dims: Optional dict mapping session_id → neuron count.

        Returns:
            Configured TeacherMamba instance.
        """
        model_cfg = config.get("model", {})
        return cls(
            input_size=input_size,
            hidden_size=model_cfg.get("hidden_size", 256),
            num_layers=model_cfg.get("num_layers", 3),
            d_state=model_cfg.get("d_state", 16),
            d_conv=model_cfg.get("d_conv", 4),
            expand=model_cfg.get("expand", 2),
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
        )
