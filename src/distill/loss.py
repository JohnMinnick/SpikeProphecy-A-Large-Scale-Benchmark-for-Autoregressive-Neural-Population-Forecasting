"""
Distillation loss function for ANN-to-SNN training.

Combines:
1. Poisson NLL (Supervised): Matches ground truth spike counts.
2. Poisson KL (Distillation): Transfers teacher's uncertainty structure
   via closed-form KL-divergence between Poisson distributions.
3. Regularization: Penalizes high firing rates (L1/L2 on spikes).

Equation:
    L = L_poisson(y, y_hat) + beta(t) * L_kl(y_teacher, y_hat) + alpha * L_reg(spikes)

Optional weight scheduling:
    beta(t) anneals from distill_weight → distill_weight_min over training epochs.
    This lets the student absorb teacher structure early, then refine on ground truth.

Channel masking (multi-session support):
    When channel_mask is provided, PoissonNLL and PoissonKL are computed
    per-element, multiplied by the mask, and averaged over unmasked elements
    only. This prevents padding neurons from contributing gradient signal
    at large m_max scales.
"""

import math
import torch
import torch.nn as nn


class DistillationLoss(nn.Module):
    """
    Composite loss for distillation with optional weight scheduling.

    Uses Poisson KL-divergence for the distillation term, which penalizes
    *relative* rate errors rather than absolute differences (as MSE would).
    This is a natural fit for Poisson-distributed spike count predictions.

    Args:
        distill_weight (beta): Initial weight for Poisson KL distillation loss.
        distill_weight_min: Minimum KL weight after annealing (default: None = no schedule).
        distill_schedule: "cosine" or "linear" annealing (default: None = fixed weight).
        reg_weight (alpha): Weight for spike regularization.
        reg_type: "l1" (sum of absolute) or "l2" (sum of squares) for spikes.
        log_input: Whether inputs to PoissonNLL are log-rates (default False).
    """

    # Numerical stability constant for log/division in KL computation
    _EPS = 1e-8

    def __init__(
        self,
        distill_weight: float = 0.5,
        distill_weight_min: float = None,
        distill_schedule: str = None,
        reg_weight: float = 0.0,
        reg_type: str = "l1",
        log_input: bool = False,
    ):
        super().__init__()
        # Initial (and possibly fixed) KL weight
        self.distill_weight_init = distill_weight
        self.distill_weight = distill_weight

        # Scheduling: if min is set, weight anneals over epochs
        self.distill_weight_min = distill_weight_min
        self.distill_schedule = distill_schedule

        self.reg_weight = reg_weight
        self.reg_type = reg_type
        self.log_input = log_input

        self.poisson = nn.PoissonNLLLoss(log_input=log_input, full=True)

    def set_epoch(self, epoch: int, total_epochs: int) -> None:
        """
        Update distill_weight based on the current epoch.

        Anneals from distill_weight_init → distill_weight_min using the
        configured schedule. No-op if scheduling is disabled.

        Args:
            epoch: Current epoch (0-indexed).
            total_epochs: Total number of training epochs.
        """
        if self.distill_weight_min is None or self.distill_schedule is None:
            return  # No scheduling — keep fixed weight

        # Fraction of training completed [0, 1]
        progress = min(epoch / max(total_epochs - 1, 1), 1.0)
        w_max = self.distill_weight_init
        w_min = self.distill_weight_min

        if self.distill_schedule == "cosine":
            # Cosine annealing: smooth decay from max → min
            self.distill_weight = w_min + 0.5 * (w_max - w_min) * (
                1 + math.cos(math.pi * progress)
            )
        elif self.distill_schedule == "linear":
            # Linear decay from max → min
            self.distill_weight = w_max + (w_min - w_max) * progress
        else:
            raise ValueError(
                f"Unknown distill_schedule '{self.distill_schedule}'. "
                f"Must be 'cosine' or 'linear'."
            )

    def _poisson_kl(
        self,
        lambda_t: torch.Tensor,
        lambda_s: torch.Tensor,
        channel_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Closed-form KL-divergence between two Poisson distributions.

        KL(Poisson(λ_t) || Poisson(λ_s)) = λ_t * log(λ_t / λ_s) - λ_t + λ_s

        Both inputs are clamped to _EPS for numerical stability.

        Args:
            lambda_t: Teacher rates (reference distribution).
            lambda_s: Student rates (approximating distribution).
            channel_mask: Optional (Batch, Out) binary mask. 1 = real neuron,
                0 = padding. If provided, KL is averaged over unmasked
                elements only.

        Returns:
            Scalar mean KL-divergence across unmasked elements.
        """
        lambda_t = lambda_t.clamp(min=self._EPS)
        lambda_s = lambda_s.clamp(min=self._EPS)
        per_element = (
            lambda_t * torch.log(lambda_t / lambda_s) - lambda_t + lambda_s
        )

        if channel_mask is not None:
            # Average over unmasked elements only
            return (
                (per_element * channel_mask).sum()
                / channel_mask.sum().clamp(min=1.0)
            )
        return per_element.mean()

    def _masked_poisson_nll(
        self,
        student_rates: torch.Tensor,
        ground_truth: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute masked Poisson NLL — only penalize real (unmasked) channels.

        Uses per-element reduction, applies channel mask, and averages
        over unmasked elements. This ensures padding neurons contribute
        zero gradient signal.

        Args:
            student_rates: Predicted rates (Batch, Out).
            ground_truth: True spike counts (Batch, Out).
            channel_mask: (Batch, Out) binary mask. 1 = real, 0 = padding.

        Returns:
            Scalar masked Poisson NLL.
        """
        # Per-element Poisson NLL: λ - y * log(λ) + log(y!)
        rates = student_rates.clamp(min=self._EPS)
        if self.log_input:
            # If log_input, student_rates are log-rates: exp(s) - y * s
            per_element = torch.exp(student_rates) - ground_truth * student_rates
        else:
            per_element = rates - ground_truth * torch.log(rates)

        # Apply mask and average over real neurons only
        return (
            (per_element * channel_mask).sum()
            / channel_mask.sum().clamp(min=1.0)
        )

    def forward(
        self,
        student_rates: torch.Tensor,
        student_spikes: torch.Tensor,
        ground_truth: torch.Tensor,
        teacher_rates: torch.Tensor,
        channel_mask: torch.Tensor = None,
    ) -> dict:
        """
        Compute total loss and components.

        Args:
            student_rates: Predicted rates (Batch, Out).
            student_spikes: Hidden spikes (Batch, T, Hidden).
            ground_truth: True spike counts (Batch, Out).
            teacher_rates: Teacher predicted rates (Batch, Out).
            channel_mask: Optional (Batch, Out) binary mask. 1 = real neuron,
                0 = padding channel. When provided, PoissonNLL and PoissonKL
                are computed only over real neurons, preventing padding from
                diluting the gradient signal.

        Returns:
             dict: {"loss": total, "poisson": ..., "distill": ..., "reg": ...,
                     "distill_weight": current weight}
        """
        # Zero out teacher rates on padding channels to eliminate
        # any residual KL signal from near-zero padding predictions
        if channel_mask is not None:
            teacher_rates = teacher_rates * channel_mask

        # Supervised loss: Poisson NLL against ground-truth counts
        if channel_mask is not None:
            loss_poisson = self._masked_poisson_nll(
                student_rates, ground_truth, channel_mask,
            )
        else:
            loss_poisson = self.poisson(student_rates, ground_truth)

        # Distillation loss: Poisson KL-divergence between teacher and student
        # rates. Both teacher and student output softplus rates (non-negative),
        # so the Poisson KL closed form applies directly.
        loss_distill = self._poisson_kl(
            teacher_rates, student_rates, channel_mask,
        )

        # Spike regularization (encourages sparse firing)
        # Note: operates on hidden spikes, not output — no masking needed
        if self.reg_weight > 0:
            if self.reg_type == "l1":
                # Mean absolute spike activity (encourages sparsity)
                loss_reg = student_spikes.sum() / student_spikes.numel()
            elif self.reg_type == "l2":
                # Mean squared spike activity
                loss_reg = (student_spikes ** 2).sum() / student_spikes.numel()
            else:
                raise ValueError(
                    f"Unknown reg_type '{self.reg_type}'. "
                    f"Must be 'l1' or 'l2'."
                )
        else:
            loss_reg = torch.tensor(0.0, device=student_rates.device)

        # Total composite loss (distill_weight may be scheduled)
        total = (
            loss_poisson
            + self.distill_weight * loss_distill
            + self.reg_weight * loss_reg
        )

        return {
            "loss": total,
            "poisson": loss_poisson,
            "distill": loss_distill,
            "reg": loss_reg,
            "distill_weight": self.distill_weight,
        }
