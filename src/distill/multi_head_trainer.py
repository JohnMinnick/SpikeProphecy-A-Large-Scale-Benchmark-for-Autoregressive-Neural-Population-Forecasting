"""
Trainer for multi-head SNN distillation with behavioral auxiliary heads.

Extends DistillTrainer to handle:
    - Student models that return dicts (with aux head outputs)
    - Behavioral data (contrast, response, trial masks) in each batch
    - MultiHeadDistillationLoss with trial-masked auxiliary losses
    - Optional teacher hidden states for hidden-state alignment
    - SGC (Smoothed Gradient Compensation) lambda annealing
"""

import logging
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.eval.metrics import (
    mae, mse, pearson_r, poisson_nll, r_squared,
    population_rate_r, spatial_pattern_r, population_cosine_sim,
)
from src.distill.distill_trainer import DistillTrainer
from src.distill.multi_head_loss import MultiHeadDistillationLoss

logger = logging.getLogger(__name__)


class MultiHeadDistillTrainer(DistillTrainer):
    """
    Trainer for multi-head distillation.

    Expects data loaders to yield either:
        (inputs, targets, teacher_rates, behavior_dict)       — 4-tuple
        (inputs, targets, teacher_rates, behavior_dict, teacher_hidden)  — 5-tuple

    where behavior_dict contains trial-level behavioral variables and
    teacher_hidden (optional) contains internal teacher hidden states
    for hidden-state alignment loss.

    SGC annealing:
        If the student model has sgc_enabled=True, the trainer anneals
        _sgc_lambda from sgc_lambda_init → 0.0 over sgc_warmdown_epochs.
    """

    def _train_one_epoch(self) -> float:
        """
        Run one training epoch with multi-head distillation.

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

        # -----------------------------------------------------------------
        # SGC lambda annealing: decay _sgc_lambda from init → 0.0 over
        # sgc_warmdown_epochs.  After warmdown, SGC is fully disabled.
        # -----------------------------------------------------------------
        if hasattr(self.model, 'sgc_enabled') and self.model.sgc_enabled:
            sgc_cfg = getattr(self, '_sgc_config', {})
            warmdown = sgc_cfg.get('warmdown_epochs', 10)
            init_lambda = sgc_cfg.get('lambda_init', 0.5)
            epoch = self._distill_epoch

            if epoch < warmdown:
                # Linear decay: init_lambda → 0.0 over warmdown epochs
                progress = epoch / max(warmdown - 1, 1)
                new_lambda = init_lambda * (1.0 - progress)
            else:
                new_lambda = 0.0

            self.model._sgc_lambda = new_lambda
            logger.info(
                "  SGC lambda=%.4f (epoch %d, warmdown=%d)",
                new_lambda, epoch + 1, warmdown,
            )

        self._distill_epoch += 1

        total_loss = 0.0
        total_stim_loss = 0.0
        total_resp_loss = 0.0
        total_align_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            # Support 6-tuple (with channel_mask), 5-tuple, and 4-tuple
            if len(batch) == 6:
                x, y, y_teacher, behavior, teacher_hidden, channel_mask = batch
            elif len(batch) == 5:
                x, y, y_teacher, behavior, teacher_hidden = batch
                channel_mask = None
            else:
                x, y, y_teacher, behavior = batch
                teacher_hidden = None
                channel_mask = None

            x = x.to(self.device)
            y = y.to(self.device)
            y_teacher = y_teacher.to(self.device)

            # Move behavior tensors to device
            behavior_dev = {
                k: v.to(self.device) for k, v in behavior.items()
            }

            # Move teacher_hidden to device if present
            teacher_hidden_dev = (
                teacher_hidden.to(self.device)
                if teacher_hidden is not None else None
            )

            # Move channel_mask to device if present
            channel_mask_dev = (
                channel_mask.to(self.device)
                if channel_mask is not None else None
            )

            # Forward pass — multi-head student returns dict
            student_output = self.model(x)

            # Handle backward-compatible tuple output
            if isinstance(student_output, tuple):
                student_output = {
                    "rates": student_output[0],
                    "spikes": student_output[1],
                }

            # Multi-head distillation loss (with optional masking + alignment)
            loss_dict = self.criterion(
                student_output, y, y_teacher, behavior_dev,
                teacher_hidden=teacher_hidden_dev,
                channel_mask=channel_mask_dev,
            )
            loss = loss_dict["loss"]

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm,
                )

            self.optimizer.step()

            total_loss += loss.item()
            total_stim_loss += loss_dict["stimulus_loss"].item()
            total_resp_loss += loss_dict["response_loss"].item()
            total_align_loss += loss_dict["hidden_align_loss"].item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Log auxiliary head losses
        if n_batches > 0:
            logger.info(
                "  Train: total=%.4f, stim=%.4f, resp=%.4f, align=%.4f",
                avg_loss,
                total_stim_loss / n_batches,
                total_resp_loss / n_batches,
                total_align_loss / n_batches,
            )

        return avg_loss

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """
        Run validation with multi-head metrics.

        Computes per-session Pearson r (matching the teacher's
        Trainer.evaluate() mask-weighted approach) by tracking session
        boundaries and excluding padding channels per-session.

        Returns:
            Dict with distillation + auxiliary head metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_poisson = 0.0
        total_distill = 0.0
        total_reg = 0.0
        total_stim_loss = 0.0
        total_resp_loss = 0.0
        total_align_loss = 0.0

        # Per-session accumulators for dynamics metrics.
        # The distillation data loader iterates session-by-session,
        # so we track session boundaries by detecting when the active
        # neuron count changes or when we see the session transition
        # pattern in the data (padding columns becoming active or
        # inactive).
        #
        # Each entry: (preds_list, targets_list, m_i)
        # where m_i = number of real (non-padded) neurons in that session.
        session_buffers = []
        current_session_preds = []
        current_session_targets = []
        current_m_i = None
        current_batch_m_i = None

        # Accumulators for auxiliary metrics
        correct_responses = 0
        total_responses = 0
        n_batches = 0

        for batch in self.val_loader:
            # Support 6-tuple (with channel_mask), 5-tuple, and 4-tuple
            if len(batch) == 6:
                x, y, y_teacher, behavior, teacher_hidden, channel_mask = batch
            elif len(batch) == 5:
                x, y, y_teacher, behavior, teacher_hidden = batch
                channel_mask = None
            else:
                x, y, y_teacher, behavior = batch
                teacher_hidden = None
                channel_mask = None

            x = x.to(self.device)
            y = y.to(self.device)
            y_teacher = y_teacher.to(self.device)
            behavior_dev = {
                k: v.to(self.device) for k, v in behavior.items()
            }
            teacher_hidden_dev = (
                teacher_hidden.to(self.device)
                if teacher_hidden is not None else None
            )
            channel_mask_dev = (
                channel_mask.to(self.device)
                if channel_mask is not None else None
            )

            # Forward
            student_output = self.model(x)
            if isinstance(student_output, tuple):
                student_output = {
                    "rates": student_output[0],
                    "spikes": student_output[1],
                }

            # Multi-head loss (with optional masking + alignment)
            loss_dict = self.criterion(
                student_output, y, y_teacher, behavior_dev,
                teacher_hidden=teacher_hidden_dev,
                channel_mask=channel_mask_dev,
            )

            total_loss += loss_dict["loss"].item()
            total_poisson += loss_dict["poisson"].item()
            total_distill += loss_dict["distill"].item()
            total_reg += loss_dict["reg"].item()
            total_stim_loss += loss_dict["stimulus_loss"].item()
            total_resp_loss += loss_dict["response_loss"].item()
            total_align_loss += loss_dict["hidden_align_loss"].item()

            # ---------------------------------------------------------
            # Track per-session predictions/targets for dynamics metrics.
            #
            # Session tracking strategy:
            #   1. Use val_loader.current_session_id if available
            #      (set by CyclingMultiSessionLoader during iteration)
            #   2. Fall back to channel_mask.sum() for stable M_i
            #      (mask is deterministic per session, unlike non-zero
            #       column counts which fluctuate with stochastic spikes)
            #   3. Last resort: count non-zero target columns (unreliable)
            # ---------------------------------------------------------
            rates_cpu = student_output["rates"].cpu()
            y_cpu = y.cpu()

            # Determine session identity
            loader_session_id = getattr(
                self.val_loader, 'current_session_id', None,
            )

            # Determine real neuron count from mask (stable) or fallback
            if channel_mask_dev is not None:
                batch_m_i = int(channel_mask_dev[0].sum().item())
            else:
                batch_m_i = int((y_cpu.abs().sum(dim=0) > 0).sum())

            # Build a composite session key for boundary detection
            session_key = loader_session_id if loader_session_id else batch_m_i

            if current_m_i is not None and session_key != current_m_i:
                # Session boundary detected — flush current session
                if current_session_preds:
                    # Determine real neuron count for slicing
                    flush_m_i = current_batch_m_i
                    session_buffers.append((
                        torch.cat(current_session_preds, dim=0),
                        torch.cat(current_session_targets, dim=0),
                        flush_m_i,
                    ))
                current_session_preds = []
                current_session_targets = []

            current_m_i = session_key
            current_batch_m_i = batch_m_i
            current_session_preds.append(rates_cpu)
            current_session_targets.append(y_cpu)

            # Response accuracy on trial-active bins
            if (
                "response" in student_output
                and "response_choice" in behavior_dev
            ):
                trial_mask = behavior_dev["trial_active"] > 0.5
                if trial_mask.sum() > 0:
                    pred_class = student_output["response"][trial_mask].argmax(dim=-1)
                    true_class = (
                        behavior_dev["response_choice"][trial_mask] + 1
                    ).long()
                    correct_responses += (pred_class == true_class).sum().item()
                    total_responses += int(trial_mask.sum().item())

            n_batches += 1

        # Flush the last session
        if current_session_preds:
            session_buffers.append((
                torch.cat(current_session_preds, dim=0),
                torch.cat(current_session_targets, dim=0),
                current_batch_m_i if current_batch_m_i else 0,
            ))

        # Average losses
        metrics = {
            "val_loss": total_loss / max(n_batches, 1),
            "val_poisson_loss": total_poisson / max(n_batches, 1),
            "val_distill_loss": total_distill / max(n_batches, 1),
            "val_reg_loss": total_reg / max(n_batches, 1),
            "val_stimulus_loss": total_stim_loss / max(n_batches, 1),
            "val_response_loss": total_resp_loss / max(n_batches, 1),
            "val_hidden_align_loss": total_align_loss / max(n_batches, 1),
        }

        # -----------------------------------------------------------------
        # Dynamics forecast metrics — per-session, then weighted average.
        #
        # Metrics are computed on the FULL m_max output (including padding
        # channels) to match the teacher's Trainer.evaluate() approach.
        # Correctly predicting zero for non-existent padding neurons is
        # a legitimate capability that should be measured.
        # -----------------------------------------------------------------
        if session_buffers:
            session_r_vals = []    # (mean_r, n_neurons) per session
            # Population metric accumulators: (value, n_neurons)
            session_pop_rate_r = []
            session_spatial_r = []
            session_cosine = []
            all_active_preds = []  # For aggregate scalar metrics
            all_active_targets = []

            for preds, targets, m_i in session_buffers:
                # Use full m_max output (including padding channels)
                # to match teacher's Trainer.evaluate() approach.
                # Correctly predicting zero for padding IS a real
                # capability and should be reflected in metrics.
                s_preds = preds
                s_targets = targets
                n_ch = preds.shape[1]

                # Per-channel Pearson r for this session
                s_r = float(pearson_r(s_preds, s_targets))
                session_r_vals.append((s_r, n_ch))

                # Population-level metrics for this session
                session_pop_rate_r.append(
                    (float(population_rate_r(s_preds, s_targets)), n_ch),
                )
                session_spatial_r.append(
                    (float(spatial_pattern_r(s_preds, s_targets)), n_ch),
                )
                session_cosine.append(
                    (float(population_cosine_sim(s_preds, s_targets)), n_ch),
                )

                # Collect for aggregate metrics
                all_active_preds.append(s_preds.reshape(-1))
                all_active_targets.append(s_targets.reshape(-1))

            # Neuron-weighted average across sessions
            total_neurons = sum(n for _, n in session_r_vals)

            def _weighted_mean(vals_and_weights):
                """Compute neuron-weighted average of per-session values."""
                total_n = sum(n for _, n in vals_and_weights)
                if total_n > 0:
                    return sum(v * n for v, n in vals_and_weights) / total_n
                return 0.0

            # --- Three complementary Pearson r methods ---
            #
            # 1. Per-channel mean (Welford-equivalent): matches teacher's
            #    Trainer.evaluate(). Each channel weighted equally.
            # 2. Activity-weighted: weights each channel by its total GT
            #    spike count, so active neurons matter more.
            # 3. Global flatten: single r over all (time × neurons),
            #    captures both spatial and temporal structure at once.
            weighted_r = _weighted_mean(session_r_vals)

            # Activity-weighted r: per-channel r weighted by GT activity
            cat_preds_2d = torch.cat(
                [p for p, _, _ in session_buffers], dim=0,
            )
            cat_targets_2d = torch.cat(
                [t for _, t, _ in session_buffers], dim=0,
            )
            per_ch_r = pearson_r(cat_preds_2d, cat_targets_2d, per_channel=True)
            activity_weights = cat_targets_2d.sum(dim=0)
            activity_weights = activity_weights / (
                activity_weights.sum() + 1e-8
            )
            activity_weighted_r = float(
                (per_ch_r * activity_weights).sum()
            )

            # Global flatten r: single correlation over all elements
            flat_preds = cat_preds_2d.reshape(-1)
            flat_targets = cat_targets_2d.reshape(-1)
            pred_c = flat_preds - flat_preds.mean()
            targ_c = flat_targets - flat_targets.mean()
            num = (pred_c * targ_c).sum()
            den = pred_c.pow(2).sum().sqrt() * targ_c.pow(2).sum().sqrt()
            global_r = float(num / den) if den > 0 else 0.0

            metrics["val_pearson_r"] = weighted_r
            metrics["val_pearson_r_weighted"] = activity_weighted_r
            metrics["val_pearson_r_global"] = global_r
            metrics["val_n_sessions"] = len(session_buffers)
            metrics["val_total_neurons"] = total_neurons

            # Population-level metrics (auto-logged to WandB)
            metrics["val_pop_rate_r"] = _weighted_mean(session_pop_rate_r)
            metrics["val_spatial_r"] = _weighted_mean(session_spatial_r)
            metrics["val_pop_cosine"] = _weighted_mean(session_cosine)

            logger.info(
                "  Per-session eval: %d sessions, %d neurons | "
                "val_r=%.4f | r_weighted=%.4f | r_global=%.4f | "
                "pop_rate_r=%.4f | spatial_r=%.4f | cosine=%.4f",
                len(session_buffers), total_neurons,
                weighted_r, activity_weighted_r, global_r,
                metrics["val_pop_rate_r"],
                metrics["val_spatial_r"],
                metrics["val_pop_cosine"],
            )

            # Aggregate scalar metrics (NLL, MAE, MSE, R^2) on
            # concatenated active-only data. These metrics don't
            # suffer from the cross-session mixing issue because
            # they're decomposable (each element contributes
            # independently, unlike Pearson r which uses means).
            cat_preds = torch.cat(all_active_preds, dim=0)
            cat_targets = torch.cat(all_active_targets, dim=0)

            metrics["val_poisson_nll"] = float(
                poisson_nll(cat_preds, cat_targets, log_input=False),
            )
            metrics["val_r_squared"] = float(
                r_squared(cat_preds, cat_targets),
            )
            metrics["val_mae"] = float(mae(cat_preds, cat_targets))
            metrics["val_mse"] = float(mse(cat_preds, cat_targets))
        else:
            # No data — return zeros
            metrics["val_pearson_r"] = 0.0
            metrics["val_poisson_nll"] = 0.0
            metrics["val_r_squared"] = 0.0
            metrics["val_mae"] = 0.0
            metrics["val_mse"] = 0.0
            metrics["val_pop_rate_r"] = 0.0
            metrics["val_spatial_r"] = 0.0
            metrics["val_pop_cosine"] = 0.0

        # Response accuracy
        if total_responses > 0:
            metrics["val_response_accuracy"] = correct_responses / total_responses
            logger.info(
                "  Val response accuracy: %.4f (%d/%d trial bins)",
                metrics["val_response_accuracy"],
                correct_responses, total_responses,
            )

        return metrics
