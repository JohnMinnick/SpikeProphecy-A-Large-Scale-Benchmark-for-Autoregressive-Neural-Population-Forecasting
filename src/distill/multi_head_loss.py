"""
Multi-head distillation loss for SNN with auxiliary behavioral heads.

Extends the base DistillationLoss with:
    - L_stimulus: CrossEntropy on 16-class contrast-pair classification
    - L_response: CrossEntropy on response choice predictions (trial-masked)

Total loss:
    L = L_dynamics + λ_stim * L_stimulus + λ_resp * L_response
      + λ_align * L_hidden_align

where L_dynamics = PoissonNLL + β * PoissonKL + α * L1_reg (from base),
and L_hidden_align = MSE(student_membrane, teacher_hidden) aligns the
student's pre-threshold membrane potentials with the teacher's internal
hidden states (dual-level distillation).

Stimulus classes map each (left_contrast, right_contrast) pair from
{0, 0.25, 0.5, 1.0} × {0, 0.25, 0.5, 1.0} → class index 0–15.

Trial masking ensures stimulus and response losses only compute over
time bins that fall within an active trial (first ~33-42% of each
Steinmetz recording session).
"""

import logging

import torch
import torch.nn as nn

from src.distill.loss import DistillationLoss

logger = logging.getLogger(__name__)

# Canonical contrast levels in the Steinmetz 2019 visual discrimination task
CONTRAST_LEVELS = [0.0, 0.25, 0.5, 1.0]


def contrast_to_class_index(
    left_contrast: torch.Tensor,
    right_contrast: torch.Tensor,
) -> torch.Tensor:
    """
    Map (left, right) contrast pair to a class index 0–15.

    Encoding: class = left_idx * 4 + right_idx
    where left_idx, right_idx ∈ {0, 1, 2, 3} correspond to
    contrast levels {0.0, 0.25, 0.5, 1.0}.

    Args:
        left_contrast: (Batch,) left visual contrast values.
        right_contrast: (Batch,) right visual contrast values.

    Returns:
        (Batch,) int64 class indices 0–15.
    """
    # Snap to nearest valid level to handle float imprecision
    levels = torch.tensor(CONTRAST_LEVELS, device=left_contrast.device)
    left_idx = torch.argmin(
        (left_contrast.unsqueeze(-1) - levels).abs(), dim=-1
    )
    right_idx = torch.argmin(
        (right_contrast.unsqueeze(-1) - levels).abs(), dim=-1
    )
    return (left_idx * 4 + right_idx).long()


class MultiHeadDistillationLoss(DistillationLoss):
    """
    Multi-head distillation loss with behavioral auxiliary losses.

    Inherits the dynamics loss (PoissonNLL + PoissonKL + spike reg)
    and adds stimulus and response losses with trial masking.

    Args:
        stimulus_weight: Weight for stimulus contrast loss (λ_stim).
        response_weight: Weight for response classification loss (λ_resp).
        hidden_align_weight: Weight for hidden-state MSE alignment (λ_align).
            Default 0.0 disables hidden alignment (backward compatible).
        **kwargs: All base DistillationLoss arguments.
    """

    def __init__(
        self,
        stimulus_weight: float = 0.1,
        response_weight: float = 0.1,
        hidden_align_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.stimulus_weight = stimulus_weight
        self.response_weight = response_weight
        self.hidden_align_weight = hidden_align_weight

        # CrossEntropy for 16-class contrast-pair classification
        self.stimulus_criterion = nn.CrossEntropyLoss(reduction="none")

        # CrossEntropy for response classification
        # reduction="none" so we can apply trial masking
        self.response_criterion = nn.CrossEntropyLoss(reduction="none")

        logger.info(
            "MultiHeadDistillationLoss: stimulus_weight=%.3f, "
            "response_weight=%.3f, hidden_align_weight=%.3f",
            stimulus_weight, response_weight, hidden_align_weight,
        )

    def forward(
        self,
        student_output: dict,
        ground_truth: torch.Tensor,
        teacher_rates: torch.Tensor,
        behavior: dict = None,
        teacher_hidden: torch.Tensor = None,
        channel_mask: torch.Tensor = None,
    ) -> dict:
        """
        Compute total multi-head loss.

        Args:
            student_output: Dict with keys:
                - "rates": Predicted spike rates (Batch, Out).
                - "spikes": Hidden spikes (Batch, T, Hidden).
                - "stimulus": Optional contrast predictions (Batch, 2).
                - "response": Optional response logits (Batch, 3).
            ground_truth: True spike counts (Batch, Out).
            teacher_rates: Teacher predicted rates (Batch, Out).
            behavior: Optional dict with keys:
            teacher_hidden: Optional teacher hidden states (Batch, T, H)
                from the last Mamba block.  Used for hidden-state alignment
                when hidden_align_weight > 0.
                - "left_contrast": (Batch,) ground truth left contrast.
                - "right_contrast": (Batch,) ground truth right contrast.
                - "response_choice": (Batch,) ground truth response (-1, 0, +1).
                - "trial_active": (Batch,) binary mask — 1 if bin in a trial.
                - "behavior_train_mask": (Batch,) optional — 1 for train
                  trials, 0 for held-out eval trials. If absent, falls
                  back to trial_active (all trials used).
            channel_mask: Optional (Batch, Out) binary mask. 1 = real neuron,
                0 = padding. Passed through to base DistillationLoss.

        Returns:
            Dict with all loss components and total loss.
        """
        # --- Dynamics loss (with optional channel masking) ---
        base_result = super().forward(
            student_output["rates"],
            student_output["spikes"],
            ground_truth,
            teacher_rates,
            channel_mask=channel_mask,
        )

        total_loss = base_result["loss"]
        result = {
            **base_result,
            "stimulus_loss": torch.tensor(0.0, device=total_loss.device),
            "response_loss": torch.tensor(0.0, device=total_loss.device),
            "hidden_align_loss": torch.tensor(0.0, device=total_loss.device),
            "n_trial_bins": 0,
        }

        # --- Hidden-state alignment (MSE between teacher hidden & student membrane) ---
        if (
            self.hidden_align_weight > 0
            and teacher_hidden is not None
            and "membrane_potentials" in student_output
        ):
            student_mem = student_output["membrane_potentials"]  # (B, T, H)
            # Truncate to shorter sequence (teacher may have different T)
            t_min = min(student_mem.shape[1], teacher_hidden.shape[1])
            align_loss = nn.functional.mse_loss(
                student_mem[:, :t_min, :],
                teacher_hidden[:, :t_min, :],
            )
            total_loss = total_loss + self.hidden_align_weight * align_loss
            result["hidden_align_loss"] = align_loss

        # --- Stimulus loss (trial-masked CE on 16-class contrast pair) ---
        if (
            behavior is not None
            and "stimulus" in student_output
            and "left_contrast" in behavior
        ):
            # Use behavior_train_mask to exclude held-out eval trials;
            # fall back to trial_active if mask not present (backward compat)
            mask_key = (
                "behavior_train_mask"
                if "behavior_train_mask" in behavior
                else "trial_active"
            )
            trial_mask = behavior[mask_key] > 0.5  # (Batch,)
            n_trial = int(trial_mask.sum().item())

            if n_trial > 0:
                # Map (left, right) contrast to class index 0–15
                stim_labels = contrast_to_class_index(
                    behavior["left_contrast"][trial_mask],
                    behavior["right_contrast"][trial_mask],
                )

                # CrossEntropy on masked trial bins
                pred_logits = student_output["stimulus"][trial_mask]  # (N, 16)
                stim_loss = self.stimulus_criterion(
                    pred_logits, stim_labels,
                ).mean()

                total_loss = total_loss + self.stimulus_weight * stim_loss
                result["stimulus_loss"] = stim_loss
                result["n_trial_bins"] = n_trial

        # --- Response loss (trial-masked CrossEntropy on choice) ---
        if (
            behavior is not None
            and "response" in student_output
            and "response_choice" in behavior
        ):
            # Use behavior_train_mask (same fallback as stimulus)
            mask_key = (
                "behavior_train_mask"
                if "behavior_train_mask" in behavior
                else "trial_active"
            )
            trial_mask = behavior[mask_key] > 0.5  # (Batch,)
            n_trial = int(trial_mask.sum().item())

            if n_trial > 0:
                # Response choice is -1, 0, +1 → map to class indices 0, 1, 2
                raw_choice = behavior["response_choice"][trial_mask]
                response_labels = (raw_choice + 1).long()  # -1→0, 0→1, +1→2

                pred_logits = student_output["response"][trial_mask]  # (N, 3)
                resp_loss = self.response_criterion(
                    pred_logits, response_labels,
                ).mean()

                total_loss = total_loss + self.response_weight * resp_loss
                result["response_loss"] = resp_loss
                result["n_trial_bins"] = max(
                    result["n_trial_bins"], n_trial,
                )

        result["loss"] = total_loss
        return result
