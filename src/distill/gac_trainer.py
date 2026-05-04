"""
Trainer for GAC-SNN mechanism-aligned distillation.

Extends DistillTrainer to use MechanismAlignmentLoss and pass
internal signals from both the instrumented Mamba teacher and
the GacStudentSNN through the mechanism alignment loss.

The key difference from standard distillation: the teacher
must be run online (not pre-cached) because we need the internal
Δ/B/C signals from the Mamba SSM's forward pass.

KOSMOS Tier 2F: Mechanism-aligned Mamba→SNN distillation.
"""

import logging
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.eval.metrics import mae, mse, pearson_r, poisson_nll
from src.train.trainer import Trainer
from src.distill.mechanism_loss import MechanismAlignmentLoss

logger = logging.getLogger(__name__)


class GacDistillTrainer(Trainer):
    """
    Trainer for GAC-SNN mechanism-aligned distillation.

    Unlike standard DistillTrainer which receives pre-computed teacher
    rates, this trainer runs the teacher online each batch to capture
    internal SSM signals (Δ, B, C) for mechanism alignment.

    Data loaders yield (x_student, y, x_teacher_padded) triplets where:
    - x_student is the unpadded session input  (batch, T, N_i)
    - y is the ground truth spike counts        (batch, N_i)
    - x_teacher_padded is M_max-padded input    (batch, T, M_max)

    Args:
        student: GacStudentSNN model.
        teacher: Instrumented TeacherMamba model (frozen).
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Combined config dict.
        device: Torch device.
        criterion: MechanismAlignmentLoss instance.
        session_id: Session ID for teacher's session-specific head.
        exp_dir: Optional experiment directory.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        criterion: MechanismAlignmentLoss,
        session_id: Optional[str] = None,
        exp_dir: Any = None,
    ):
        # Initialize with student model
        super().__init__(
            student, train_loader, val_loader, config, device, exp_dir,
        )
        # Replace parent's criterion with mechanism alignment loss
        self.criterion = criterion.to(device)
        self.teacher = teacher
        self.session_id = session_id

        # BUG FIX: The parent __init__ creates an optimizer with only
        # self.model.parameters(). The MechanismAlignmentLoss has
        # learnable SignalProjectors that MUST be in the optimizer.
        # Re-create the optimizer with both student + criterion params.
        training_cfg = config.get("training", {})
        lr = training_cfg.get("learning_rate", 1e-3)
        wd = training_cfg.get("weight_decay", 0.0)
        all_params = list(self.model.parameters()) + list(
            self.criterion.parameters()
        )
        self.optimizer = torch.optim.AdamW(
            all_params, lr=lr, weight_decay=wd,
        )
        logger.info(
            "Optimizer includes %d student + %d projector params",
            sum(p.numel() for p in self.model.parameters()),
            sum(p.numel() for p in self.criterion.parameters()),
        )

        # Epoch tracking for distill_weight scheduling
        self._distill_epoch = 0
        self._total_epochs = config.get("training", {}).get("epochs", 50)

        # --- Staged training: alignment warmup + rampup ---
        # During warmup, gamma_tau/stp/dend are set to 0 so the SNN
        # trains purely on Poisson NLL + KL distillation. After warmup,
        # the gammas linearly ramp up over rampup_epochs to their full
        # config values. This prevents alignment losses from fighting
        # the untrained base model.
        distill_cfg = config.get("distillation", {})
        self._alignment_warmup = distill_cfg.get(
            "alignment_warmup_epochs", 0,
        )
        self._alignment_rampup = distill_cfg.get(
            "alignment_rampup_epochs", 1,
        )
        # Store the full/target gamma values from config
        self._gamma_tau_full = self.criterion.gamma_tau
        self._gamma_stp_full = self.criterion.gamma_stp
        self._gamma_dend_full = self.criterion.gamma_dend
        logger.info(
            "Staged alignment: warmup=%d epochs, rampup=%d epochs, "
            "full gammas: τ=%.4f, STP=%.4f, dend=%.4f",
            self._alignment_warmup, self._alignment_rampup,
            self._gamma_tau_full, self._gamma_stp_full,
            self._gamma_dend_full,
        )

        # Optional warmup bypass: skip STP+dendrite during warmup
        # so the base SNN trains like a standard RSynaptic model.
        self._warmup_bypass = distill_cfg.get(
            "warmup_bypass", False,
        )
        if self._warmup_bypass:
            logger.info(
                "Warmup bypass ENABLED: STP+dendrite disabled during "
                "warmup (%d epochs)", self._alignment_warmup,
            )

    def _compute_alignment_scale(self, epoch: int) -> float:
        """
        Compute the alignment loss scaling factor for the current epoch.

        Returns 0.0 during warmup, linearly ramps from 0→1 during
        rampup, and returns 1.0 after warmup+rampup.

        Args:
            epoch: Current epoch (0-indexed).

        Returns:
            Scale factor in [0.0, 1.0].
        """
        if epoch < self._alignment_warmup:
            return 0.0
        rampup_progress = epoch - self._alignment_warmup
        if rampup_progress >= self._alignment_rampup:
            return 1.0
        return rampup_progress / max(self._alignment_rampup, 1)

    def _train_one_epoch(self) -> float:
        """
        Run one training epoch with mechanism-aligned distillation.

        Handles batch formats from different loaders:
        - 2-tuple (x, y): single-session or multi-session shared-head
        - 3-tuple (x, y, mask): multi-session with channel masking
        - 3-tuple (x, y, x_teacher): legacy single-session GAC loader

        For multi-session shared-head mode (no explicit x_teacher),
        uses x for both student and teacher input.

        Each batch:
        1. Teacher forward (frozen, instrumented) → rates + Δ/B/C
        2. Student forward → rates + spikes + alignment signals
        3. MechanismAlignmentLoss combines all 6 loss terms
        4. Backward through student + projectors only

        Returns:
            Mean total training loss.
        """
        self.model.train()
        self.teacher.eval()

        # Update distill_weight schedule
        if hasattr(self.criterion, 'set_epoch'):
            self.criterion.set_epoch(
                self._distill_epoch, self._total_epochs,
            )

        # --- Staged alignment: scale gammas based on warmup/rampup ---
        align_scale = self._compute_alignment_scale(self._distill_epoch)
        self.criterion.gamma_tau = self._gamma_tau_full * align_scale
        self.criterion.gamma_stp = self._gamma_stp_full * align_scale
        self.criterion.gamma_dend = self._gamma_dend_full * align_scale

        logger.info(
            "  epoch %d/%d: distill_w=%.4f, align_scale=%.3f "
            "(γ_τ=%.4f, γ_stp=%.4f, γ_dend=%.4f)",
            self._distill_epoch + 1, self._total_epochs,
            self.criterion.distill_weight, align_scale,
            self.criterion.gamma_tau, self.criterion.gamma_stp,
            self.criterion.gamma_dend,
        )

        # --- Warmup bypass: disable STP+dendrite during warmup ---
        if self._warmup_bypass and hasattr(self.model, 'set_warmup_mode'):
            in_warmup = self._distill_epoch < self._alignment_warmup
            self.model.set_warmup_mode(in_warmup)
            if in_warmup:
                logger.info(
                    "  warmup bypass: STP+dendrite DISABLED (epoch %d/%d)",
                    self._distill_epoch + 1, self._alignment_warmup,
                )

        self._distill_epoch += 1

        total_loss = 0.0
        total_tau = 0.0
        total_stp = 0.0
        total_dend = 0.0
        n_batches = 0

        for batch in self.train_loader:
            # Unpack batch — supports multiple formats:
            # (x, y) — no mask, no separate teacher input
            # (x, y, mask) — multi-session with channel mask
            # (x, y, x_teacher) — legacy single-session GAC
            if len(batch) == 3:
                x, y, third = batch
                # Distinguish mask from x_teacher by shape:
                # mask is (batch, M) with values in {0, 1}
                # x_teacher is (batch, T, M) with 3 dims
                if third.dim() == 2:
                    # 2D → channel mask (batch, M)
                    mask = third
                    x_teacher = x  # Use x for both
                else:
                    # 3D → x_teacher (batch, T, M)
                    mask = None
                    x_teacher = third
            elif len(batch) == 2:
                x, y = batch
                mask = None
                x_teacher = x  # Use x for both
            else:
                x, y = batch[0], batch[1]
                mask = None
                x_teacher = x

            x = x.to(self.device)
            y = y.to(self.device)
            x_teacher = x_teacher.to(self.device)
            if mask is not None:
                mask = mask.to(self.device)

            # ---- Teacher forward (frozen + instrumented) ----
            with torch.no_grad():
                if self.session_id is not None:
                    teacher_rates = self.teacher(
                        x_teacher, session_id=self.session_id,
                    )
                else:
                    teacher_rates = self.teacher(x_teacher)

                # Get internal Mamba signals (Δ, B, C)
                mamba_signals = self.teacher.get_internal_signals()

            # ---- Student forward ----
            student_rates, student_spikes = self.model(x)

            # Get SNN alignment signals (β, STP gains, dendrite gates)
            snn_signals = self.model.get_alignment_signals()

            # ---- Mechanism alignment loss ----
            loss_dict = self.criterion(
                student_rates=student_rates,
                student_spikes=student_spikes,
                ground_truth=y,
                teacher_rates=teacher_rates.detach(),
                mamba_signals=mamba_signals,
                snn_signals=snn_signals,
                mask=mask,
            )
            loss = loss_dict["loss"]

            # ---- Backward through student + projectors ----
            self.optimizer.zero_grad()
            loss.backward()

            if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                # Clip student + projector gradients
                all_params = list(self.model.parameters())
                all_params += list(self.criterion.parameters())
                nn.utils.clip_grad_norm_(all_params, self.grad_clip_norm)

            self.optimizer.step()

            total_loss += loss.item()
            total_tau += loss_dict.get("tau_align", torch.tensor(0.0)).item()
            total_stp += loss_dict.get("stp_align", torch.tensor(0.0)).item()
            total_dend += loss_dict.get(
                "dend_align", torch.tensor(0.0),
            ).item()
            n_batches += 1

        # Log alignment loss breakdown
        if n_batches > 0:
            logger.info(
                "  align: τ=%.4f  STP=%.4f  dend=%.4f",
                total_tau / n_batches,
                total_stp / n_batches,
                total_dend / n_batches,
            )

        return total_loss / max(n_batches, 1)

    # NOTE: _validate is intentionally NOT overridden here.
    # The parent Trainer._validate() → evaluate() uses:
    #   - Mask-aware per-channel Pearson r (excludes zero-padded channels)
    #   - Float64 Welford streaming for numerical stability
    #   - Per-session sufficient statistics for variable output sizes
    # This is critical for correct multi-session metrics.
    # Previously this was overridden with a broken naive implementation
    # that computed pearson_r over ALL channels (including ~1140 zeros),
    # producing artificially low val_r (0.215 vs expected ~0.40+).
