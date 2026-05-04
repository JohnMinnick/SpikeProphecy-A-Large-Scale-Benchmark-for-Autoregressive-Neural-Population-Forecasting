"""
Trainer for Student SNN distillation.

Inherits from the standard Trainer but overrides the training step to handle
teacher targets and spiking regularization.
"""

import logging
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.eval.metrics import mae, mse, pearson_r, poisson_nll
from src.train.trainer import Trainer
from src.distill.loss import DistillationLoss

logger = logging.getLogger(__name__)


class DistillTrainer(Trainer):
    """
    Trainer for distillation.

    Expects data loaders to yield (inputs, targets, teacher_rates).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        criterion: DistillationLoss,
        exp_dir: Any = None,
    ):
        super().__init__(
            model, train_loader, val_loader, config, device, exp_dir
        )
        # Override the parent's PoissonNLLLoss with our composite
        # distillation loss and move it to the target device.
        self.criterion = criterion.to(device)

        # Epoch tracking for distill_weight scheduling
        self._distill_epoch = 0
        self._total_epochs = config.get("training", {}).get("epochs", 50)

    def _train_one_epoch(self) -> float:
        """
        Run one training epoch with distillation.

        Returns:
            Mean total training loss.
        """
        self.model.train()

        # Update distill_weight schedule (no-op if scheduling disabled)
        if hasattr(self.criterion, 'set_epoch'):
            self.criterion.set_epoch(self._distill_epoch, self._total_epochs)
            logger.info(
                "  distill_weight=%.4f (epoch %d/%d)",
                self.criterion.distill_weight,
                self._distill_epoch + 1,
                self._total_epochs,
            )
        self._distill_epoch += 1

        total_loss = 0.0
        n_batches = 0

        for x, y, y_teacher in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            y_teacher = y_teacher.to(self.device)

            # Forward pass — student returns (rates, spikes)
            rates, spikes = self.model(x)

            # Composite distillation loss
            loss_dict = self.criterion(rates, spikes, y, y_teacher)
            loss = loss_dict["loss"]

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """
        Run validation with distillation metrics.

        We override the parent because our data loaders yield triplets
        (inputs, targets, teacher_rates) instead of pairs.

        Returns:
            Dict with distillation losses and standard forecast metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_poisson = 0.0
        total_distill = 0.0
        total_reg = 0.0

        # Accumulators for standard metrics
        targets_list = []
        preds_list = []
        n_batches = 0

        for x, y, y_teacher in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            y_teacher = y_teacher.to(self.device)

            # Forward
            rates, spikes = self.model(x)

            # Composite loss
            loss_dict = self.criterion(rates, spikes, y, y_teacher)

            total_loss += loss_dict["loss"].item()
            total_poisson += loss_dict["poisson"].item()
            total_distill += loss_dict["distill"].item()
            total_reg += loss_dict["reg"].item()

            # Collect predictions/targets on CPU for aggregate metrics
            targets_list.append(y.cpu())
            preds_list.append(rates.cpu())
            n_batches += 1

        # Average losses across batches
        metrics = {
            "val_loss": total_loss / max(n_batches, 1),
            "val_poisson_loss": total_poisson / max(n_batches, 1),
            "val_distill_loss": total_distill / max(n_batches, 1),
            "val_reg_loss": total_reg / max(n_batches, 1),
        }

        # Standard forecast metrics (Pearson r, MAE, MSE, Poisson NLL)
        # computed directly on tensors to avoid numpy round-trip
        all_preds = torch.cat(preds_list, dim=0)
        all_targets = torch.cat(targets_list, dim=0)

        metrics["val_poisson_nll"] = float(
            poisson_nll(all_preds, all_targets, log_input=False)
        )
        metrics["val_pearson_r"] = float(pearson_r(all_preds, all_targets))
        metrics["val_mae"] = float(mae(all_preds, all_targets))
        metrics["val_mse"] = float(mse(all_preds, all_targets))

        return metrics
