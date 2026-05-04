"""
DEPRECATED — FGL (Future-Guided Learning) Trainer.

⚠️ This module is LEGACY. FGL underperformed standard distillation (val_r=0.40
vs 0.42) and was superseded by DistillTrainer. See ADR-0014 for history.

Manages the two-model training loop where a frozen teacher provides
soft targets (via Poisson KL divergence) and a student learns to
predict further ahead using only causal inputs.

The composite loss is:
    L_FGL = alpha * PoissonNLL(student, y_true) + (1 - alpha) * PoissonKL(teacher, student)

Both teacher and student are ANNs (LRU v2). The student trains to
approximate the teacher's predictions despite having less temporal context.
"""

import logging
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.eval.metrics import mae, mse, pearson_r, poisson_nll
from src.train.trainer import Trainer

logger = logging.getLogger(__name__)

# Numerical stability constant for Poisson KL
_EPS = 1e-8


class FGLTrainer(Trainer):
    """
    Trainer for Future-Guided Learning.

    Expects DataLoaders yielding (x_student, x_teacher, y_target) triplets.
    The teacher model is frozen (no gradients); only the student is trained.

    Args:
        teacher: Pretrained teacher model (frozen, eval mode).
        student: Student model to train.
        train_loader: FGL DataLoader yielding triplets.
        val_loader: FGL DataLoader yielding triplets.
        config: Training config dict.
        device: Torch device.
        alpha: Weight for supervised loss (1-alpha for distillation).
        exp_dir: Experiment directory for saving checkpoints.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        alpha: float = 0.5,
        exp_dir: Any = None,
    ):
        # Initialize parent Trainer with the student model
        super().__init__(
            student, train_loader, val_loader, config, device, exp_dir,
        )

        # Freeze the teacher — no gradients, eval mode permanently
        self.teacher = teacher.to(device)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.alpha = alpha
        self.poisson_nll_loss = nn.PoissonNLLLoss(
            log_input=False, full=True,
        )

        logger.info(
            "FGLTrainer initialized: alpha=%.2f, teacher frozen (%d params)",
            alpha,
            sum(p.numel() for p in self.teacher.parameters()),
        )

    def _poisson_kl(
        self,
        lambda_t: torch.Tensor,
        lambda_s: torch.Tensor,
    ) -> torch.Tensor:
        """
        Closed-form Poisson KL divergence: KL(P_teacher || P_student).

        KL = lambda_t * log(lambda_t / lambda_s) - lambda_t + lambda_s

        Args:
            lambda_t: Teacher predicted rates (non-negative).
            lambda_s: Student predicted rates (non-negative).

        Returns:
            Scalar mean KL divergence.
        """
        lambda_t = lambda_t.clamp(min=_EPS)
        lambda_s = lambda_s.clamp(min=_EPS)
        kl = lambda_t * (torch.log(lambda_t) - torch.log(lambda_s)) \
            - lambda_t + lambda_s
        return kl.mean()

    def _train_one_epoch(self) -> float:
        """
        Run one FGL training epoch.

        For each batch:
          1. Forward teacher on x_teacher (no grad) → lambda_teacher
          2. Forward student on x_student → lambda_student
          3. Compute composite loss: alpha * PoissonNLL + (1-alpha) * PoissonKL
          4. Backprop through student only

        Returns:
            Mean composite training loss.
        """
        self.model.train()  # Student
        total_loss = 0.0
        n_batches = 0

        for x_student, x_teacher, y_target in self.train_loader:
            x_student = x_student.to(self.device)
            x_teacher = x_teacher.to(self.device)
            y_target = y_target.to(self.device)

            # Teacher forward (frozen, no grad)
            with torch.no_grad():
                teacher_out = self.teacher(x_teacher)
                # Handle both (rates,) and (rates, hidden) returns
                lambda_teacher = (
                    teacher_out[0] if isinstance(teacher_out, tuple)
                    else teacher_out
                )

            # Student forward
            student_out = self.model(x_student)
            lambda_student = (
                student_out[0] if isinstance(student_out, tuple)
                else student_out
            )

            # Composite FGL loss
            loss_supervised = self.poisson_nll_loss(lambda_student, y_target)
            loss_distill = self._poisson_kl(lambda_teacher, lambda_student)
            loss = self.alpha * loss_supervised + (1 - self.alpha) * loss_distill

            # Backward (student only)
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm,
                )

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """
        Run FGL validation with both supervised and distillation metrics.

        Returns:
            Dict with val_loss, val_supervised, val_distill, val_pearson_r, etc.
        """
        self.model.eval()
        total_loss = 0.0
        total_supervised = 0.0
        total_distill = 0.0

        preds_list = []
        targets_list = []
        n_batches = 0

        for x_student, x_teacher, y_target in self.val_loader:
            x_student = x_student.to(self.device)
            x_teacher = x_teacher.to(self.device)
            y_target = y_target.to(self.device)

            # Teacher forward
            teacher_out = self.teacher(x_teacher)
            lambda_teacher = (
                teacher_out[0] if isinstance(teacher_out, tuple)
                else teacher_out
            )

            # Student forward
            student_out = self.model(x_student)
            lambda_student = (
                student_out[0] if isinstance(student_out, tuple)
                else student_out
            )

            # Losses
            loss_supervised = self.poisson_nll_loss(lambda_student, y_target)
            loss_distill = self._poisson_kl(lambda_teacher, lambda_student)
            loss = self.alpha * loss_supervised + (1 - self.alpha) * loss_distill

            total_loss += loss.item()
            total_supervised += loss_supervised.item()
            total_distill += loss_distill.item()

            preds_list.append(lambda_student.cpu())
            targets_list.append(y_target.cpu())
            n_batches += 1

        # Aggregate forecast metrics
        all_preds = torch.cat(preds_list, dim=0)
        all_targets = torch.cat(targets_list, dim=0)

        metrics = {
            "val_loss": total_loss / max(n_batches, 1),
            "val_supervised_loss": total_supervised / max(n_batches, 1),
            "val_distill_loss": total_distill / max(n_batches, 1),
            "val_pearson_r": float(pearson_r(all_preds, all_targets)),
            "val_poisson_nll": float(
                poisson_nll(all_preds, all_targets, log_input=False)
            ),
            "val_mae": float(mae(all_preds, all_targets)),
            "val_mse": float(mse(all_preds, all_targets)),
        }

        return metrics
