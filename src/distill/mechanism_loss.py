"""
Mechanism Alignment Loss for GAC-SNN distillation.

Extends the standard DistillationLoss with three auxiliary alignment
terms that match the SNN's internal dynamics to the Mamba teacher's
input-dependent signals:

  L_total = L_supervised + β·L_KL
          + γ_τ·L_tau     (Δ ↔ τ alignment)
          + γ_B·L_stp     (B ↔ STP alignment)
          + γ_C·L_dendrite (C ↔ dendrite alignment)
          + α·L_reg       (spike regularization)

Each alignment term uses a small projection network (Linear → LayerNorm
→ sigmoid) to map between Mamba's signal space and the SNN's signal
space, since the representations have different dimensionalities and
distributions.

KOSMOS Tier 2F: Mechanism-aligned Mamba→SNN distillation.
"""

import logging
import math
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SignalProjector(nn.Module):
    """
    Small projector network to map between Mamba and SNN signal spaces.

    Maps a Mamba signal (e.g., Δ_t from dt_proj) to the SNN's
    corresponding signal space (e.g., β_t from SelectiveRSynaptic)
    so they can be compared via MSE.

    Architecture: Linear → LayerNorm → Hardtanh(0, 1)

    NOTE: Previously used Sigmoid, which caused gradient saturation on
    large-magnitude teacher signals. Hardtanh clips to [0, 1] without
    the vanishing gradient problem in saturated regions.

    Args:
        input_dim: Mamba signal dimension (d_inner or similar).
        output_dim: SNN signal dimension (hidden_size).
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Hardtanh(min_val=0.0, max_val=1.0),
        )
        # Initialize near identity if dims match, small otherwise
        nn.init.normal_(self.proj[0].weight, std=0.01)
        nn.init.zeros_(self.proj[0].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project Mamba signal to SNN signal space.

        Args:
            x: Mamba signal tensor.

        Returns:
            Projected signal in [0, 1].
        """
        return self.proj(x)


class MechanismAlignmentLoss(nn.Module):
    """
    Composite loss for mechanism-aligned GAC-SNN distillation.

    Combines:
    1. Poisson NLL (supervised ground truth)
    2. Poisson KL (standard distillation from teacher rates)
    3. τ-alignment: MSE(β_SNN, proj(Δ_Mamba))
    4. STP-alignment: MSE(stp_gain, proj(B_Mamba))
    5. Dendrite-alignment: MSE(dend_gate, proj(C_Mamba))
    6. Spike regularization

    Alignment projectors are learned simultaneously with the SNN,
    initialized near zero so alignment starts soft and strengthens
    as the projectors find the right mapping.

    Args:
        d_delta: Mamba Δ signal dimension (d_inner = d_model * expand).
        d_state: Mamba B/C signal dimension (SSM state dimension).
        snn_hidden_size: SNN hidden size.
        distill_weight: KL distillation weight (β).
        gamma_tau: τ-alignment weight (default 0.1).
        gamma_stp: STP-alignment weight (default 0.1).
        gamma_dend: Dendrite-alignment weight (default 0.1).
        reg_weight: Spike regularization weight (α).
        reg_type: "l1" or "l2".
        distill_weight_min: Minimum KL weight (for scheduling).
        distill_schedule: "cosine" or "linear" annealing.
    """

    _EPS = 1e-8

    def __init__(
        self,
        d_delta: int,
        d_state: int,
        snn_hidden_size: int,
        distill_weight: float = 0.5,
        gamma_tau: float = 0.1,
        gamma_stp: float = 0.1,
        gamma_dend: float = 0.1,
        reg_weight: float = 0.0,
        reg_type: str = "l1",
        distill_weight_min: Optional[float] = None,
        distill_schedule: Optional[str] = None,
        log_input: bool = False,
    ):
        super().__init__()

        # Standard distillation components
        self.distill_weight_init = distill_weight
        self.distill_weight = distill_weight
        self.distill_weight_min = distill_weight_min
        self.distill_schedule = distill_schedule
        self.reg_weight = reg_weight
        self.reg_type = reg_type
        self.poisson = nn.PoissonNLLLoss(log_input=log_input, full=True)

        # Mechanism alignment weights
        self.gamma_tau = gamma_tau
        self.gamma_stp = gamma_stp
        self.gamma_dend = gamma_dend

        # Projector networks: Mamba space → SNN space
        # Each signal has its own dimension: Δ is d_inner, B/C are d_state
        if gamma_tau > 0:
            self.tau_projector = SignalProjector(
                d_delta, snn_hidden_size,
            )
            logger.info(
                "τ-alignment projector: %d → %d (γ_τ=%.3f)",
                d_delta, snn_hidden_size, gamma_tau,
            )
        else:
            self.tau_projector = None

        if gamma_stp > 0:
            self.stp_projector = SignalProjector(
                d_state, snn_hidden_size,
            )
            logger.info(
                "STP-alignment projector: %d → %d (γ_B=%.3f)",
                d_state, snn_hidden_size, gamma_stp,
            )
        else:
            self.stp_projector = None

        if gamma_dend > 0:
            self.dend_projector = SignalProjector(
                d_state, snn_hidden_size,
            )
            logger.info(
                "Dendrite-alignment projector: %d → %d (γ_C=%.3f)",
                d_state, snn_hidden_size, gamma_dend,
            )
        else:
            self.dend_projector = None

    def set_epoch(self, epoch: int, total_epochs: int) -> None:
        """Update distill_weight based on the current epoch."""
        if self.distill_weight_min is None or self.distill_schedule is None:
            return

        progress = min(epoch / max(total_epochs - 1, 1), 1.0)
        w_max = self.distill_weight_init
        w_min = self.distill_weight_min

        if self.distill_schedule == "cosine":
            self.distill_weight = w_min + 0.5 * (w_max - w_min) * (
                1 + math.cos(math.pi * progress)
            )
        elif self.distill_schedule == "linear":
            self.distill_weight = w_max + (w_min - w_max) * progress

    def _poisson_kl(
        self,
        lambda_t: torch.Tensor,
        lambda_s: torch.Tensor,
    ) -> torch.Tensor:
        """Closed-form KL(Poisson(λ_t) || Poisson(λ_s))."""
        lambda_t = lambda_t.clamp(min=self._EPS)
        lambda_s = lambda_s.clamp(min=self._EPS)
        return (
            lambda_t * torch.log(lambda_t / lambda_s) - lambda_t + lambda_s
        ).mean()

    def _compute_alignment_loss(
        self,
        projector: Optional[SignalProjector],
        mamba_signal: Optional[torch.Tensor],
        snn_signal: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute per-timestep alignment MSE between projected Mamba and SNN
        signals.

        CRITICAL: Does NOT average across time before comparing. Instead,
        aligns signals at each timestep independently, then averages the
        per-timestep MSE values. This preserves the temporal dynamics
        that are the whole point of mechanism alignment.

        Handles shape differences:
        - Mamba signals: (B*T, d) from x_proj hooks, or (B, T, d)
        - SNN signals:   (B, T, H) from forward pass, or (B, H)

        The projector maps Mamba features → SNN feature space per timestep.

        Returns zero if either signal or projector is None.
        """
        if projector is None or mamba_signal is None or snn_signal is None:
            return torch.tensor(0.0, device=(
                snn_signal.device if snn_signal is not None
                else torch.device("cpu")
            ))

        # --- Normalize SNN signal to 3D (B, T, H) ---
        if snn_signal.dim() == 2:
            # (B, H) — no temporal dim (e.g., dendrite gates)
            # Expand to (B, 1, H) for consistent handling
            snn_3d = snn_signal.unsqueeze(1)
        elif snn_signal.dim() == 3:
            snn_3d = snn_signal  # Already (B, T, H)
        else:
            snn_3d = snn_signal.view(-1, 1, snn_signal.shape[-1])

        B, T_snn, H = snn_3d.shape

        # --- Normalize Mamba signal to 3D (B, T, d) ---
        if mamba_signal.dim() == 2:
            # (N, d) — could be (B*T, d) from hooks or (B, d)
            N, d = mamba_signal.shape
            if N == B * T_snn:
                # Reshape from (B*T, d) → (B, T, d)
                mamba_3d = mamba_signal.view(B, T_snn, d)
            elif N == B:
                # Already (B, d) — expand to (B, 1, d)
                mamba_3d = mamba_signal.unsqueeze(1)
            elif N % B == 0:
                # (B*T', d) where T' != T_snn — reshape then
                # interpolate or average to match T_snn
                T_mamba = N // B
                mamba_3d = mamba_signal.view(B, T_mamba, d)
                if T_mamba != T_snn:
                    # Adaptive average pooling to match timesteps
                    # (B, T_mamba, d) → (B, d, T_mamba) → pool → (B, d, T_snn)
                    mamba_3d = mamba_3d.permute(0, 2, 1)
                    mamba_3d = nn.functional.adaptive_avg_pool1d(
                        mamba_3d, T_snn,
                    )
                    mamba_3d = mamba_3d.permute(0, 2, 1)
            else:
                # Fallback: truncate to B and expand
                mamba_3d = mamba_signal[:B].unsqueeze(1)
        elif mamba_signal.dim() == 3:
            mamba_3d = mamba_signal  # Already (B, T, d)
            if mamba_3d.shape[1] != T_snn:
                # Temporal mismatch — pool to match
                mamba_3d = mamba_3d.permute(0, 2, 1)
                mamba_3d = nn.functional.adaptive_avg_pool1d(
                    mamba_3d, T_snn,
                )
                mamba_3d = mamba_3d.permute(0, 2, 1)
        else:
            mamba_3d = mamba_signal.view(B, -1, mamba_signal.shape[-1])

        # --- Project per-timestep: (B, T, d) → (B, T, H) ---
        # Flatten to (B*T, d), project, reshape back
        BT = mamba_3d.shape[0] * mamba_3d.shape[1]
        d = mamba_3d.shape[2]
        projected = projector(mamba_3d.reshape(BT, d))  # (B*T, H)
        projected = projected.view(B, mamba_3d.shape[1], -1)  # (B, T, H)

        # --- Truncate feature dim if needed ---
        min_dim = min(projected.shape[-1], snn_3d.shape[-1])
        projected = projected[..., :min_dim]
        snn_target = snn_3d[..., :min_dim]

        # --- Per-timestep MSE, averaged over (B, T, H) ---
        return nn.functional.mse_loss(projected, snn_target.detach())

    def forward(
        self,
        student_rates: torch.Tensor,
        student_spikes: torch.Tensor,
        ground_truth: torch.Tensor,
        teacher_rates: torch.Tensor,
        mamba_signals: Optional[Dict[str, torch.Tensor]] = None,
        snn_signals: Optional[Dict[str, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute total mechanism-aligned loss.

        Args:
            student_rates: Predicted rates (batch, out).
            student_spikes: Hidden spikes (batch, T, hidden).
            ground_truth: True spike counts (batch, out).
            teacher_rates: Teacher predicted rates (batch, out).
            mamba_signals: Dict from teacher.get_internal_signals():
                - 'delta': (batch, T, d_inner) Δ projections
                - 'B': (batch, T, d_inner) B projections
                - 'C': (batch, T, d_inner) C projections
            snn_signals: Dict from student.get_alignment_signals():
                - 'betas': (batch, T, hidden) selective decay
                - 'stp_gains': (batch, T, hidden) STP gains
                - 'dendrite_gates': (batch, hidden) dendritic gates
            mask: Optional binary channel mask (batch, out). 1=real, 0=padded.
                  When provided, Poisson NLL and KL losses only compute
                  over unmasked channels (critical for multi-session).

        Returns:
            Dict with all loss components.
        """
        device = student_rates.device

        # Standard supervised loss (mask-aware)
        if mask is not None:
            # Per-element Poisson NLL, masked and averaged
            eps = self._EPS
            per_elem = student_rates - ground_truth * torch.log(
                student_rates + eps
            )
            loss_poisson = (per_elem * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            loss_poisson = self.poisson(student_rates, ground_truth)

        # Standard KL distillation (mask-aware)
        if mask is not None:
            lambda_t = teacher_rates.clamp(min=self._EPS)
            lambda_s = student_rates.clamp(min=self._EPS)
            per_elem_kl = (
                lambda_t * torch.log(lambda_t / lambda_s)
                - lambda_t + lambda_s
            )
            loss_distill = (
                (per_elem_kl * mask).sum() / mask.sum().clamp(min=1.0)
            )
        else:
            loss_distill = self._poisson_kl(teacher_rates, student_rates)

        # Mechanism alignment losses
        if mamba_signals is None:
            mamba_signals = {}
        if snn_signals is None:
            snn_signals = {}

        # τ-alignment: Mamba Δ ↔ SNN β (selective decay)
        loss_tau = self._compute_alignment_loss(
            self.tau_projector,
            mamba_signals.get("delta"),
            snn_signals.get("betas"),
        ).to(device)

        # STP-alignment: Mamba B ↔ SNN STP gain
        loss_stp = self._compute_alignment_loss(
            self.stp_projector,
            mamba_signals.get("B"),
            snn_signals.get("stp_gains"),
        ).to(device)

        # Dendrite-alignment: Mamba C ↔ SNN dendritic gate
        loss_dend = self._compute_alignment_loss(
            self.dend_projector,
            mamba_signals.get("C"),
            snn_signals.get("dendrite_gates"),
        ).to(device)

        # Spike regularization
        if self.reg_weight > 0:
            if self.reg_type == "l1":
                loss_reg = student_spikes.sum() / student_spikes.numel()
            else:
                loss_reg = (
                    student_spikes ** 2
                ).sum() / student_spikes.numel()
        else:
            loss_reg = torch.tensor(0.0, device=device)

        # Total loss
        total = (
            loss_poisson
            + self.distill_weight * loss_distill
            + self.gamma_tau * loss_tau
            + self.gamma_stp * loss_stp
            + self.gamma_dend * loss_dend
            + self.reg_weight * loss_reg
        )

        return {
            "loss": total,
            "poisson": loss_poisson,
            "distill": loss_distill,
            "tau_align": loss_tau,
            "stp_align": loss_stp,
            "dend_align": loss_dend,
            "reg": loss_reg,
            "distill_weight": self.distill_weight,
        }
