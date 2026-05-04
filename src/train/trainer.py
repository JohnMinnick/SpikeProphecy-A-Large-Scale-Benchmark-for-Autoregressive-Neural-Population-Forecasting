"""
Training engine for the teacher ANN.

Provides a Trainer class that handles:
    - Training loop with configurable loss (Poisson NLL, NegBin NLL, ZIP NLL)
    - Validation with all metrics (NLL, Pearson r, MAE, MSE)
    - AdamW optimizer with cosine LR scheduling + step-based linear warmup
    - Early stopping on validation loss
    - Model checkpointing (best + final)
    - Per-epoch metric logging

Usage:
    from src.train.trainer import Trainer
    trainer = Trainer(model, train_loader, val_loader, config, device)
    history = trainer.train()
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    MultiStepLR,
)
from torch.utils.data import DataLoader

from src.eval.metrics import (
    mae, mse, pearson_r, poisson_nll,
    negative_binomial_nll, zero_inflated_poisson_nll,
)
from src.eval.ceiling_weights import build_ceiling_weights
from src.eval.comparison_metrics import (
    bits_per_spike as bps_metric,
    pearson_r_per_neuron as r_per_neuron_np,
)
from src.eval.empirical_ceiling import ceiling_efficiency
from src.train.cmp_loss import cmp_nll, cmp_nll_per_element, LearnableDispersion

logger = logging.getLogger(__name__)

# Valid loss function types (includes distillation for GAC-SNN)
VALID_LOSS_TYPES = (
    "poisson_nll", "negbin_nll", "zip_nll", "cmp_nll",
    "region_hybrid", "fano_adaptive", "mechanism_alignment",
)

# Lazy W&B import — only activated when wandb is installed
# AND WANDB_API_KEY is set (i.e., on NRP).  Zero overhead locally.
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class Trainer:
    """
    Training engine for the teacher ANN.

    Handles the full train/validate/checkpoint loop. Metrics are logged
    per epoch and stored in a history dict for post-hoc analysis.

    Args:
        model: The teacher model (must output non-negative rates).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Training config dict (from configs/teacher/default.yaml).
        device: Torch device for computation.
        exp_dir: Optional experiment directory for saving checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        exp_dir: Optional[Union[str, Path]] = None,
        checkpoint_callback: Optional[Any] = None,
        metrics_callback: Optional[Any] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.exp_dir = Path(exp_dir) if exp_dir else None
        # Optional callback invoked after best_model.pt is saved.
        # Signature: callback(checkpoint_path: Path, epoch: int)
        # Used by NRP to upload checkpoints to S3 between epochs.
        self.checkpoint_callback = checkpoint_callback
        # Optional callback invoked after each validation pass.
        # Signature: callback(epoch: int, history: dict)
        # Used by NRP to upload metrics.json to S3 incrementally.
        self.metrics_callback = metrics_callback

        # Optional per-channel region map: {channel_idx: region_name}
        # When set, evaluate() computes per-region Pearson r metrics.
        # Set externally after construction (e.g., from train_teacher.py).
        self.region_map: Optional[Dict[int, str]] = None

        # Optional per-channel Fano factors: np.ndarray of shape (M,)
        # When set, evaluate() computes Fano-stratified Pearson r:
        #   sub-Poisson (FF<1), near-Poisson (1≤FF≤1.5), super-Poisson (FF>1.5)
        # Set externally after construction.
        # KOSMOS recommendation: report per-neuron r stratified by Fano.
        self.fano_factors: Optional[Any] = None

        # Optional per-channel empirical ceilings: np.ndarray of shape (M,)
        # When set, evaluate() computes ceiling efficiency (r / r_ceiling).
        # KOSMOS recommendation #1: replace Fano ceilings with empirical.
        self.empirical_ceilings: Optional[Any] = None

        # Optional behavioral trial data for PSTH R² computation.
        # Dict with keys: trial_index (T,), response_choice (T,), etc.
        # from behavior_loader.extract_trial_stimuli().
        # When set, evaluate_population() also computes PSTH R².
        # Set externally after construction (e.g., from train_teacher.py).
        self.behavior_data: Optional[Dict[str, Any]] = None

        # Optional torch.compile() for GPU kernel fusion.
        # Fuses multiple small CUDA kernels into fewer, larger ones,
        # reducing kernel launch overhead by 10-20%.  Only safe on Linux
        # (Windows torch.compile is unstable as of PyTorch 2.x).
        compute_cfg = config.get("compute", {})
        import platform
        if (
            compute_cfg.get("compile_model", False)
            and platform.system() != "Windows"
            and hasattr(torch, "compile")
        ):
            logger.info("Applying torch.compile() to model...")
            self.model = torch.compile(self.model)

        # Extract training config
        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 100)
        self.patience = train_cfg.get("patience", 15)
        self.grad_clip_norm = train_cfg.get("grad_clip_norm", 1.0)
        self.val_every_n = train_cfg.get("val_every_n", 1)
        self.start_epoch = 1  # updated by load_checkpoint() for resume

        # Extract loss config
        loss_cfg = config.get("loss", {})
        self.log_input = loss_cfg.get("log_input", False)
        self.loss_type = loss_cfg.get("type", "poisson_nll")

        # Validate loss type
        if self.loss_type not in VALID_LOSS_TYPES:
            raise ValueError(
                f"loss.type must be one of {VALID_LOSS_TYPES}, "
                f"got '{self.loss_type}'"
            )

        # --- Ceiling-based loss weighting (Priority #1 eval improvement) ---
        # Scales per-neuron loss by predictability ceiling so the model
        # focuses gradient on neurons with real rate modulation.
        # Config: loss.ceiling_weights.{enabled, stats_path, strategy, ...}
        cw_cfg = loss_cfg.get("ceiling_weights", {})
        if cw_cfg.get("enabled", False):
            cw_path = cw_cfg.get(
                "stats_path", "outputs/eval_analysis/per_neuron_stats.json"
            )
            cw_strategy = cw_cfg.get("strategy", "binary")
            cw_floor = cw_cfg.get("floor_weight", 0.1)
            cw_threshold = cw_cfg.get("threshold", 0.1)
            # m_max is inferred from model output size
            m_max = config.get("model", {}).get("output_size", None)
            if m_max is None:
                raise ValueError(
                    "loss.ceiling_weights.enabled=true requires "
                    "model.output_size to be set in config."
                )
            self.ceiling_weights = build_ceiling_weights(
                stats_path=cw_path,
                m_max=m_max,
                strategy=cw_strategy,
                floor_weight=cw_floor,
                threshold=cw_threshold,
            ).to(device)
            logger.info(
                "Ceiling weights loaded: strategy=%s, shape=%s",
                cw_strategy, self.ceiling_weights.shape,
            )
        else:
            self.ceiling_weights = None

        # --- CMP dispersion parameter (Tier 2E — KOSMOS recommendation) ---
        # When loss.type='cmp_nll', initialize a learnable per-neuron
        # dispersion parameter ν.  Starts at ν=1 (Poisson) and learns to
        # discriminate sub-Poisson (ν>1) from super-Poisson (ν<1) neurons.
        self.cmp_dispersion = None
        if self.loss_type == "cmp_nll":
            m_max = config.get("model", {}).get("output_size", None)
            if m_max is None:
                raise ValueError(
                    "loss.type='cmp_nll' requires model.output_size "
                    "to be set in config."
                )
            self.cmp_dispersion = LearnableDispersion(m_max).to(device)
            logger.info(
                "CMP dispersion initialized: %d neurons, ν₀=1.0 (Poisson)",
                m_max,
            )

        # Setup optimizer — supports per-param-group LR for selective gating.
        # When gate_lr_multiplier is set, beta_gate params get boosted LR
        # to counteract the sigmoid gradient squeeze (σ'(2.2) = 0.09).
        lr = train_cfg.get("learning_rate", 1e-3)
        weight_decay = train_cfg.get("weight_decay", 1e-4)
        gate_lr_mult = train_cfg.get("gate_lr_multiplier", 1.0)

        # Collect all trainable parameters (model + optional CMP dispersion)
        all_params = list(model.parameters())
        if self.cmp_dispersion is not None:
            all_params += list(self.cmp_dispersion.parameters())

        if gate_lr_mult != 1.0:
            # Split params: beta_gate gets boosted LR, everything else normal
            gate_params = []
            base_params = []
            for name, param in model.named_parameters():
                if "beta_gate" in name:
                    gate_params.append(param)
                else:
                    base_params.append(param)
            param_groups = [
                {"params": base_params, "lr": lr, "weight_decay": weight_decay},
                {"params": gate_params, "lr": lr * gate_lr_mult,
                 "weight_decay": weight_decay},
            ]
            # Add CMP dispersion to base group if present
            if self.cmp_dispersion is not None:
                param_groups[0]["params"] += list(
                    self.cmp_dispersion.parameters()
                )
            self.optimizer = AdamW(param_groups)
            logger.info(
                "Optimizer: 2 param groups — base_lr=%.1e, gate_lr=%.1e (%.1fx)",
                lr, lr * gate_lr_mult, gate_lr_mult,
            )
        else:
            self.optimizer = AdamW(
                all_params, lr=lr, weight_decay=weight_decay
            )

        # Step-based linear warmup: LR ramps 0 -> base_lr over N steps,
        # then epoch-level cosine decay takes over.  Step-based warmup
        # is more stable than epoch-based for multi-session training
        # where epoch length varies dramatically across sessions.
        self.base_lr = lr
        self.warmup_steps = train_cfg.get("warmup_steps", 1000)
        self.global_step = 0  # incremented each optimizer step

        # Scheduler type and warm-restart parameters
        self.scheduler_type = train_cfg.get("scheduler", "cosine")
        self.scheduler_t0 = train_cfg.get("scheduler_t0", 10)
        self.scheduler_t_mult = train_cfg.get("scheduler_t_mult", 2)
        # MultiStepLR milestones (epochs to drop LR by 10x)
        self.scheduler_milestones = train_cfg.get("scheduler_milestones", [])

        # Build LR scheduler (epoch-level, stepped after warmup finishes)
        self.scheduler = self._build_scheduler(self.epochs)

        # Mixed-precision (bf16) — enabled by default on CUDA devices.
        # bf16 is safe without GradScaler (unlike fp16) and halves
        # activation memory, which helps with large multi-session runs.
        self.use_amp = train_cfg.get("use_amp", True) and device.type == "cuda"
        self.amp_dtype = torch.bfloat16

        # Early stopping state
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        # History: per-epoch metrics for plotting
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_poisson_nll": [],
            "val_pearson_r": [],
            "val_mae": [],
            "val_mse": [],
            "learning_rate": [],
        }

        # Build scheduler description for logging
        sched_desc = self.scheduler_type
        if self.scheduler_type == "cosine_restarts":
            sched_desc += f"(T0={self.scheduler_t0}, Tmult={self.scheduler_t_mult})"

        logger.info(
            "Trainer initialized: epochs=%d, lr=%.1e, patience=%d, "
            "grad_clip=%.1f, loss=%s, device=%s, amp=%s, val_every_n=%d, "
            "warmup_steps=%d, scheduler=%s",
            self.epochs, lr, self.patience, self.grad_clip_norm,
            self.loss_type, device, self.amp_dtype if self.use_amp else "off",
            self.val_every_n, self.warmup_steps, sched_desc,
        )

        # --- W&B integration (zero-overhead when disabled) ---
        # Activates only when: wandb is installed, WANDB_API_KEY is set,
        # and we have an experiment directory.  All logging is per-epoch
        # to avoid adding overhead to the training step.
        self.use_wandb = (
            _WANDB_AVAILABLE
            and os.environ.get("WANDB_API_KEY")
            and self.exp_dir is not None
        )
        if self.use_wandb:
            # Determine run name from experiment directory
            run_name = self.exp_dir.name if self.exp_dir else None
            wandb.init(
                project="spike-prophecy",
                name=run_name,
                config=config,
                dir=str(self.exp_dir) if self.exp_dir else None,
                reinit=True,
            )
            logger.info("W&B logging enabled (project=spike-prophecy, run=%s)", run_name)

    def _build_scheduler(
        self, total_epochs: int
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Build LR scheduler based on config.

        Warmup is handled per-step in _train_one_epoch(), so this
        scheduler only manages the post-warmup decay/restarts.

        Supported scheduler types:
            - "cosine": Standard cosine annealing to 0 over total_epochs.
            - "cosine_restarts": Cosine annealing with warm restarts
              (LR periodically resets to base_lr). Restart period
              grows by T_mult each cycle.
            - "none": No scheduler (constant LR after warmup).

        Args:
            total_epochs: Total training epochs.

        Returns:
            LR scheduler instance, or None if scheduler is disabled.
        """
        if self.scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer, T_max=max(total_epochs, 1)
            )
        elif self.scheduler_type == "cosine_restarts":
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.scheduler_t0,
                T_mult=self.scheduler_t_mult,
            )
        elif self.scheduler_type == "multistep":
            # Flat-then-anneal: hold LR flat, drop at milestones.
            # Default milestones at 66% and 86% of total epochs.
            milestones = self.scheduler_milestones
            if not milestones:
                milestones = [
                    int(total_epochs * 0.66),
                    int(total_epochs * 0.86),
                ]
            return MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=0.1,  # Drop by 10x at each milestone
            )
        elif self.scheduler_type == "none":
            return None
        else:
            logger.warning(
                "Unknown scheduler '%s', falling back to cosine",
                self.scheduler_type,
            )
            return CosineAnnealingLR(
                self.optimizer, T_max=max(total_epochs, 1)
            )

    def _apply_warmup_lr(self) -> None:
        """
        Apply linear warmup LR based on current global step.

        During warmup (global_step < warmup_steps), LR is scaled
        linearly from 0 to base_lr.  After warmup completes, this
        is a no-op — the cosine scheduler takes over.
        """
        if self.warmup_steps <= 0 or self.global_step >= self.warmup_steps:
            return
        # Linear ramp: LR = base_lr * (step / warmup_steps)
        warmup_lr = self.base_lr * (self.global_step / self.warmup_steps)
        for pg in self.optimizer.param_groups:
            pg["lr"] = warmup_lr

    def _compute_loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the training loss based on the configured loss type.

        For NegBin/ZIP, also retrieves auxiliary model outputs (dispersion
        or gate) via model.get_aux_output().

        If a mask is provided, per-element loss is weighted by the mask
        and averaged over unmasked elements only (multi-session support).

        Handles shape mismatches from session-specific heads: when the model
        outputs (batch, N_i) but targets/mask are padded to (batch, M_max),
        slices targets and mask to match the model output width.

        Args:
            y_hat: Predicted rates from the model, shape (batch, M) or (batch, N_i).
            y: Ground-truth spike counts, shape (batch, M).
            mask: Optional binary mask, shape (batch, M). 1 = real channel,
                  0 = padded channel. If None, standard unmasked loss.

        Returns:
            Scalar loss tensor.
        """
        # --- Session-specific head shape alignment ---
        # When session-specific heads are active, y_hat has shape (batch, N_i)
        # but y and mask are padded to (batch, M_max). Slice to match.
        out_dim = y_hat.shape[-1]
        ceiling_w = self.ceiling_weights
        if y.shape[-1] != out_dim:
            y = y[:, :out_dim]
            if mask is not None:
                mask = mask[:, :out_dim]
            # Slice ceiling weights to match session output dim
            if ceiling_w is not None:
                ceiling_w = ceiling_w[:out_dim]

        if mask is not None:
            # Masked loss: compute per-element, apply mask, mean over unmasked
            return self._compute_masked_loss(y_hat, y, mask, ceiling_w)

        # Standard (unmasked) loss — original behavior
        if self.loss_type == "poisson_nll":
            return poisson_nll(y_hat, y, log_input=self.log_input)
        elif self.loss_type == "negbin_nll":
            # Retrieve dispersion parameter from model's auxiliary output
            aux = self.model.get_aux_output()
            if aux is None:
                raise RuntimeError(
                    "loss.type='negbin_nll' requires model.output_distribution='negbin'. "
                    "The model did not produce a dispersion output."
                )
            return negative_binomial_nll(y_hat, aux, y)
        elif self.loss_type == "zip_nll":
            # Retrieve gate parameter from model's auxiliary output
            aux = self.model.get_aux_output()
            if aux is None:
                raise RuntimeError(
                    "loss.type='zip_nll' requires model.output_distribution='zip'. "
                    "The model did not produce a gate output."
                )
            return zero_inflated_poisson_nll(y_hat, aux, y)
        elif self.loss_type == "cmp_nll":
            # CMP loss with learnable per-neuron dispersion (Tier 2E)
            nu = self.cmp_dispersion()  # (M_max,)
            return cmp_nll(y_hat, y, nu)
        elif self.loss_type in ("region_hybrid", "fano_adaptive"):
            # Region-specific or Fano-adaptive loss (KOSMOS Batch 3)
            # Requires self.region_loss_fn to be set externally
            if not hasattr(self, "region_loss_fn") or self.region_loss_fn is None:
                raise RuntimeError(
                    f"loss.type='{self.loss_type}' requires setting "
                    "trainer.region_loss_fn before training."
                )
            aux = None
            if hasattr(self.model, "get_aux_output"):
                aux = self.model.get_aux_output()
            if aux is not None:
                return self.region_loss_fn(y_hat, y, aux=aux)
            return self.region_loss_fn(y_hat, y)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _compute_masked_loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        ceiling_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute masked loss: only penalize real (non-padded) channels.

        Computes per-element loss based on the configured loss type,
        optionally applies ceiling-based per-channel reweighting,
        applies the channel mask, and averages over unmasked elements.

        Supports Poisson NLL, Negative Binomial NLL, and ZIP NLL.

        Args:
            y_hat: Predicted rates, shape (batch, M).
            y: Ground-truth counts, shape (batch, M).
            mask: Binary mask, shape (batch, M).
            ceiling_weights: Optional per-channel weight tensor, shape (M,).
                If provided, scales per-element loss before masking.

        Returns:
            Scalar masked loss.
        """
        eps = 1e-8

        if self.loss_type == "poisson_nll":
            # Per-element Poisson NLL (no reduction)
            per_element = y_hat - y * torch.log(y_hat + eps)
        elif self.loss_type == "negbin_nll":
            # Per-element Negative Binomial NLL
            aux = self.model.get_aux_output()
            if aux is None:
                raise RuntimeError(
                    "loss.type='negbin_nll' requires model.output_distribution='negbin'. "
                    "The model did not produce a dispersion output."
                )
            # NegBin NLL: -log P(y | mu, r)
            # = -[lgamma(y + r) - lgamma(r) - lgamma(y + 1)
            #     + r * log(r / (r + mu)) + y * log(mu / (r + mu))]
            r = aux.clamp(min=eps)
            mu = y_hat.clamp(min=eps)
            per_element = -(
                torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1)
                + r * torch.log(r / (r + mu))
                + y * torch.log(mu / (r + mu))
            )
        elif self.loss_type == "zip_nll":
            # Per-element Zero-Inflated Poisson NLL
            aux = self.model.get_aux_output()
            if aux is None:
                raise RuntimeError(
                    "loss.type='zip_nll' requires model.output_distribution='zip'. "
                    "The model did not produce a gate output."
                )
            gate = aux  # Probability of extra zero (pi)
            mu = y_hat.clamp(min=eps)
            # Compute log-probability under ZIP
            is_zero = (y == 0).float()
            # P(y=0) = pi + (1-pi) * exp(-mu)
            log_p_zero = torch.log(gate + (1 - gate) * torch.exp(-mu) + eps)
            # P(y>0) = (1-pi) * Poisson(y; mu)
            log_p_nonzero = (
                torch.log(1 - gate + eps) + y * torch.log(mu) - mu
                - torch.lgamma(y + 1)
            )
            # Combine: -log P(y)
            per_element = -(is_zero * log_p_zero + (1 - is_zero) * log_p_nonzero)
        elif self.loss_type == "cmp_nll":
            # Per-element CMP NLL with learnable per-neuron dispersion
            nu = self.cmp_dispersion()  # (M_max,)
            per_element = cmp_nll_per_element(y_hat, y, nu)
        elif self.loss_type in ("region_hybrid", "fano_adaptive"):
            # Region-specific or Fano-adaptive loss (KOSMOS Batch 3)
            if not hasattr(self, "region_loss_fn") or self.region_loss_fn is None:
                raise RuntimeError(
                    f"loss.type='{self.loss_type}' requires setting "
                    "trainer.region_loss_fn before training."
                )
            return self.region_loss_fn(y_hat, y, mask=mask)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Apply ceiling-based loss weighting (if configured).
        # Scales each neuron's loss by its predictability ceiling,
        # focusing gradient signal on neurons with real rate modulation.
        if ceiling_weights is None:
            ceiling_weights = self.ceiling_weights
        if ceiling_weights is not None:
            per_element = per_element * ceiling_weights.unsqueeze(0)

        # Apply mask and average over unmasked elements only
        masked_loss = (per_element * mask).sum()
        n_unmasked = mask.sum().clamp(min=1.0)
        return masked_loss / n_unmasked

    def _train_one_epoch(self) -> float:
        """
        Run one training epoch.

        Supports standard (x, y), masked (x, y, mask), and covariate
        (x, y, mask, covariates) batches.
        Applies per-step linear warmup during the first warmup_steps
        optimizer steps.

        Returns:
            Mean training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            # Unpack batch: (x, y), (x, y, mask), or (x, y, mask, covariates)
            if len(batch) == 4:
                x, y, mask, covariates = batch
                mask = mask.to(self.device)
                covariates = covariates.to(self.device)
            elif len(batch) == 3:
                x, y, mask = batch
                mask = mask.to(self.device)
                covariates = None
            else:
                x, y = batch
                mask = None
                covariates = None

            # Move to device: x=(batch, T, M), y=(batch, M)
            x = x.to(self.device)
            y = y.to(self.device)

            # Read session_id from cycling loader (None for standard loaders)
            session_id = getattr(self.train_loader, 'current_session_id', None)

            # Forward pass + loss under bf16 autocast (if enabled)
            with torch.autocast(
                self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                y_hat = self.model(
                    x, covariates=covariates, session_id=session_id,
                )
                # Unpack tuple returns (e.g. SNN returns (rates, spikes))
                if isinstance(y_hat, tuple):
                    y_hat = y_hat[0]
                loss = self._compute_loss(y_hat, y, mask=mask)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )

            self.optimizer.step()
            self.global_step += 1

            # Apply step-based linear warmup (overrides LR during warmup)
            self._apply_warmup_lr()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(
        self, loader: DataLoader, prefix: str = "eval"
    ) -> Dict[str, float]:
        """
        Run inference on an arbitrary DataLoader and return metrics.

        Uses **streaming / online metrics** to avoid accumulating all
        predictions in memory — critical for lazy per-session loading
        where the total number of samples can be in the millions.

        Decomposable metrics (loss, Poisson NLL, MAE, MSE) are computed
        as weighted batch averages.  Pearson r uses Welford-style
        sufficient statistics (sum_x, sum_y, sum_xy, sum_x², sum_y², N)
        for exact online computation.

        Args:
            loader: DataLoader to evaluate on.
            prefix: String prefix for the returned metric keys
                    (e.g. "train", "val", "test").

        Returns:
            Dict with keys ``{prefix}_loss``, ``{prefix}_poisson_nll``,
            ``{prefix}_pearson_r``, ``{prefix}_mae``, ``{prefix}_mse``.
        """
        self.model.eval()

        # Running accumulators for decomposable metrics
        total_loss = 0.0
        total_pnll = 0.0
        total_mae = 0.0
        total_mse = 0.0
        total_weight = 0.0  # Sum of mask weights (or sample counts)

        # Welford-style sufficient statistics for Pearson r
        # We accumulate per-channel sums and then compute a weighted
        # average across channels at the end.
        # For masked multi-session data, we weight by the mask.
        # Shape: (M,) accumulators, initialized lazily on first batch
        #
        # IMPORTANT: Use float64 to prevent catastrophic cancellation.
        # The formula N*Σxy - Σx*Σy computes a small difference between
        # very large numbers (~200K samples × 1240 channels). Float32
        # doesn't have enough precision when sums exceed ~10M, causing
        # garbage values (trillions, negative trillions) on larger models.
        sum_x = None   # Sum of predictions per channel
        sum_y = None   # Sum of targets per channel
        sum_xy = None  # Sum of pred * target per channel
        sum_x2 = None  # Sum of pred^2 per channel
        sum_y2 = None  # Sum of target^2 per channel
        ch_n = None    # Count of unmasked samples per channel
        ch_mask_weight = None  # Accumulated mask weight per channel

        # Session-specific heads produce variable output sizes across
        # batches. We accumulate per-session sufficient statistics
        # (keyed by session_id or output dim) so we can compute exact
        # per-session Pearson r and then average across sessions.
        prev_m = None              # Track previous batch's channel count
        variable_m = False         # True once we detect a size change
        # Per-session accumulators: {session_key: {sum_x, sum_y, ...}}
        session_stats = {}

        # Population-level metric accumulators:
        # Collect per-batch population rate sums for pop_rate_r,
        # and accumulate spatial pattern / cosine stats.
        pop_pred_rates = []   # (batch,) per-batch population rate sums
        pop_gt_rates = []     # (batch,) per-batch GT population rate sums
        spatial_r_sum = 0.0   # Running sum of per-bin spatial r
        spatial_r_count = 0   # Number of valid bins for spatial r
        cosine_sum = 0.0      # Running sum of per-bin cosine sim
        cosine_count = 0      # Number of valid bins for cosine sim

        eps = 1e-8

        for batch in loader:
            # Unpack: (x, y), (x, y, mask), or (x, y, mask, covariates)
            if len(batch) == 4:
                x, y, batch_mask, covariates = batch
                batch_mask = batch_mask.to(self.device)
                covariates = covariates.to(self.device)
            elif len(batch) == 3:
                x, y, batch_mask = batch
                batch_mask = batch_mask.to(self.device)
                covariates = None
            else:
                x, y = batch
                batch_mask = None
                covariates = None

            x = x.to(self.device)
            y = y.to(self.device)

            # Read session_id from cycling loader (None for standard loaders)
            session_id = getattr(loader, 'current_session_id', None)

            # Forward pass under bf16 autocast (if enabled)
            with torch.autocast(
                self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                y_hat = self.model(
                    x, covariates=covariates, session_id=session_id,
                )

            # Unpack tuple returns (e.g. SNN returns (rates, spikes))
            if isinstance(y_hat, tuple):
                y_hat = y_hat[0]

            # Metrics computed in float32 for numerical stability
            y_hat = y_hat.float()

            # --- Session-specific head shape alignment ---
            # When session-specific heads are active, y_hat is (batch, N_i)
            # but y and batch_mask are padded to (batch, M_max). Slice to match.
            out_dim = y_hat.shape[-1]
            if y.shape[-1] != out_dim:
                y = y[:, :out_dim]
                if batch_mask is not None:
                    batch_mask = batch_mask[:, :out_dim]

            b = y.shape[0]
            m = y.shape[1]

            # --- Initialize accumulators on first batch ---
            if sum_x is None and not variable_m:
                sum_x = torch.zeros(m, dtype=torch.float64, device=self.device)
                sum_y = torch.zeros(m, dtype=torch.float64, device=self.device)
                sum_xy = torch.zeros(m, dtype=torch.float64, device=self.device)
                sum_x2 = torch.zeros(m, dtype=torch.float64, device=self.device)
                sum_y2 = torch.zeros(m, dtype=torch.float64, device=self.device)
                ch_n = torch.zeros(m, dtype=torch.float64, device=self.device)
                ch_mask_weight = torch.zeros(m, dtype=torch.float64, device=self.device)

            # Detect if output size changed (session-specific heads)
            if prev_m is not None and m != prev_m:
                variable_m = True
                # Discard per-channel accumulators — they're now invalid
                sum_x = sum_y = sum_xy = sum_x2 = sum_y2 = None
                ch_n = ch_mask_weight = None
            prev_m = m

            # --- Compute batch weight ---
            if batch_mask is not None:
                # Masked: weight = sum of mask entries
                w = float(batch_mask.sum())
            else:
                # Unmasked: weight = batch_size * num_channels
                w = float(b * m)

            if w == 0:
                continue

            # --- Primary loss ---
            if batch_mask is not None:
                batch_loss = float(
                    self._compute_masked_loss(y_hat, y, batch_mask)
                ) * w
            elif self.loss_type == "poisson_nll":
                batch_loss = float(
                    poisson_nll(y_hat, y, log_input=self.log_input)
                ) * w
            else:
                batch_loss = float(
                    poisson_nll(y_hat, y, log_input=self.log_input)
                ) * w

            # --- Poisson NLL (always computed for comparability) ---
            per_elem_pnll = y_hat - y * torch.log(y_hat + eps)
            if batch_mask is not None:
                batch_pnll = float((per_elem_pnll * batch_mask).sum())
            else:
                batch_pnll = float(per_elem_pnll.sum())

            # --- MAE and MSE ---
            diff = y_hat - y
            if batch_mask is not None:
                batch_mae = float((diff.abs() * batch_mask).sum())
                batch_mse = float((diff.pow(2) * batch_mask).sum())
            else:
                batch_mae = float(diff.abs().sum())
                batch_mse = float(diff.pow(2).sum())

            # --- Pearson r sufficient statistics (in float64) ---
            y_hat_d = y_hat.double()  # Cast to float64 for accumulation
            y_d = y.double()

            if variable_m:
                # Variable output sizes: accumulate per-session sufficient
                # statistics so we can compute exact per-session Pearson r.
                # Use session_id as key, fallback to output dim if unavailable.
                sess_key = session_id if session_id is not None else f"m_{m}"

                if sess_key not in session_stats:
                    session_stats[sess_key] = {
                        "sum_x": torch.zeros(m, dtype=torch.float64, device=self.device),
                        "sum_y": torch.zeros(m, dtype=torch.float64, device=self.device),
                        "sum_xy": torch.zeros(m, dtype=torch.float64, device=self.device),
                        "sum_x2": torch.zeros(m, dtype=torch.float64, device=self.device),
                        "sum_y2": torch.zeros(m, dtype=torch.float64, device=self.device),
                        "ch_n": torch.zeros(m, dtype=torch.float64, device=self.device),
                        "ch_mask_weight": torch.zeros(m, dtype=torch.float64, device=self.device),
                    }

                ss = session_stats[sess_key]
                if batch_mask is not None:
                    mask_d = batch_mask.double()
                    mask_sum = mask_d.sum(dim=0)
                    ss["sum_x"] += (y_hat_d * mask_d).sum(dim=0)
                    ss["sum_y"] += (y_d * mask_d).sum(dim=0)
                    ss["sum_xy"] += (y_hat_d * y_d * mask_d).sum(dim=0)
                    ss["sum_x2"] += (y_hat_d.pow(2) * mask_d).sum(dim=0)
                    ss["sum_y2"] += (y_d.pow(2) * mask_d).sum(dim=0)
                    ss["ch_n"] += mask_sum
                    ss["ch_mask_weight"] += mask_sum
                else:
                    ss["sum_x"] += y_hat_d.sum(dim=0)
                    ss["sum_y"] += y_d.sum(dim=0)
                    ss["sum_xy"] += (y_hat_d * y_d).sum(dim=0)
                    ss["sum_x2"] += y_hat_d.pow(2).sum(dim=0)
                    ss["sum_y2"] += y_d.pow(2).sum(dim=0)
                    ss["ch_n"] += b
                    ss["ch_mask_weight"] += b
            else:
                # Fixed output size: use per-channel sufficient statistics
                if batch_mask is not None:
                    mask_d = batch_mask.double()
                    mask_sum = mask_d.sum(dim=0)  # (M,)
                    sum_x += (y_hat_d * mask_d).sum(dim=0)
                    sum_y += (y_d * mask_d).sum(dim=0)
                    sum_xy += (y_hat_d * y_d * mask_d).sum(dim=0)
                    sum_x2 += (y_hat_d.pow(2) * mask_d).sum(dim=0)
                    sum_y2 += (y_d.pow(2) * mask_d).sum(dim=0)
                    ch_n += mask_sum
                    ch_mask_weight += mask_sum
                else:
                    sum_x += y_hat_d.sum(dim=0)
                    sum_y += y_d.sum(dim=0)
                    sum_xy += (y_hat_d * y_d).sum(dim=0)
                    sum_x2 += y_hat_d.pow(2).sum(dim=0)
                    sum_y2 += y_d.pow(2).sum(dim=0)
                    ch_n += b
                    ch_mask_weight += b

            # --- Population-level metric accumulators ---
            # Population rate: sum across neurons per timestep
            if batch_mask is not None:
                pred_pop = (y_hat * batch_mask).sum(dim=-1)   # (B,)
                gt_pop = (y * batch_mask).sum(dim=-1)         # (B,)
            else:
                pred_pop = y_hat.sum(dim=-1)   # (B,)
                gt_pop = y.sum(dim=-1)         # (B,)
            pop_pred_rates.append(pred_pop.cpu())
            pop_gt_rates.append(gt_pop.cpu())

            # Spatial pattern r: per-bin correlation of neuron vectors
            pred_c = y_hat - y_hat.mean(dim=-1, keepdim=True)
            targ_c = y - y.mean(dim=-1, keepdim=True)
            sp_num = (pred_c * targ_c).sum(dim=-1)
            sp_denom = (
                pred_c.pow(2).sum(dim=-1).sqrt()
                * targ_c.pow(2).sum(dim=-1).sqrt()
            )
            sp_valid = sp_denom > 0
            if sp_valid.sum() > 0:
                spatial_r_sum += float(
                    (sp_num[sp_valid] / sp_denom[sp_valid]).sum()
                )
                spatial_r_count += int(sp_valid.sum())

            # Cosine similarity: per-bin
            p_norm = y_hat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            t_norm = y.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            cos_vals = (y_hat / p_norm * y / t_norm).sum(dim=-1)
            cosine_sum += float(cos_vals.sum())
            cosine_count += int(cos_vals.shape[0])

            # Accumulate scalar totals
            total_loss += batch_loss
            total_pnll += batch_pnll
            total_mae += batch_mae
            total_mse += batch_mse
            total_weight += w

        # --- Finalize metrics ---
        if total_weight == 0:
            return {
                f"{prefix}_loss": 0.0,
                f"{prefix}_poisson_nll": 0.0,
                f"{prefix}_pearson_r": 0.0,
                f"{prefix}_mae": 0.0,
                f"{prefix}_mse": 0.0,
            }

        loss = total_loss / total_weight
        pnll = total_pnll / total_weight
        m_val = total_mae / total_weight
        ms_val = total_mse / total_weight

        # --- Compute Pearson r ---
        if variable_m:
            # Variable output sizes: compute per-session Pearson r from
            # accumulated sufficient statistics, then average across sessions
            # weighted by total neuron-samples per session.
            session_r_vals = []  # (r_val, total_neuron_samples) per session
            for sess_key, ss in session_stats.items():
                s_ch_n = ss["ch_n"].clamp(min=1.0)
                s_cov = s_ch_n * ss["sum_xy"] - ss["sum_x"] * ss["sum_y"]
                s_var_x = (s_ch_n * ss["sum_x2"] - ss["sum_x"].pow(2)).clamp(min=0)
                s_var_y = (s_ch_n * ss["sum_y2"] - ss["sum_y"].pow(2)).clamp(min=0)
                s_denom = (s_var_x * s_var_y).sqrt().clamp(min=eps)
                s_per_ch_r = s_cov / s_denom  # Per-channel r for this session

                # Weighted average across channels (weight by usage)
                s_ch_w = ss["ch_mask_weight"]
                s_w_sum = s_ch_w.sum().clamp(min=1.0)
                s_ch_weight = s_ch_w / s_w_sum
                s_r = float((s_per_ch_r * s_ch_weight).sum())
                session_r_vals.append((s_r, float(s_w_sum)))

            if session_r_vals:
                total_r_weight = sum(w for _, w in session_r_vals)
                r_val = sum(r * w for r, w in session_r_vals) / max(total_r_weight, eps)
            else:
                r_val = 0.0
            per_ch_r = None  # Not available in variable-M mode
        else:
            # Fixed output size: compute from sufficient statistics
            n = ch_n.clamp(min=1.0)
            cov = ch_n * sum_xy - sum_x * sum_y
            var_x = (ch_n * sum_x2 - sum_x.pow(2)).clamp(min=0.0)
            var_y = (ch_n * sum_y2 - sum_y.pow(2)).clamp(min=0.0)
            denom = (var_x * var_y).sqrt().clamp(min=eps)
            per_ch_r = cov / denom  # (M,)

            # Weighted average Pearson r (weight by channel usage)
            ch_weight = ch_mask_weight / ch_mask_weight.sum().clamp(min=1.0)
            r_val = float((per_ch_r * ch_weight).sum())

        metrics = {
            f"{prefix}_loss": loss,
            f"{prefix}_poisson_nll": pnll,
            f"{prefix}_pearson_r": r_val,
            f"{prefix}_mae": m_val,
            f"{prefix}_mse": ms_val,
        }

        # --- Population-level metrics ---
        if pop_pred_rates:
            all_pred_pop = torch.cat(pop_pred_rates)
            all_gt_pop = torch.cat(pop_gt_rates)
            metrics[f"{prefix}_pop_rate_r"] = float(
                pearson_r(all_pred_pop, all_gt_pop)
            )
        else:
            metrics[f"{prefix}_pop_rate_r"] = 0.0

        metrics[f"{prefix}_spatial_r"] = (
            spatial_r_sum / spatial_r_count if spatial_r_count > 0 else 0.0
        )
        metrics[f"{prefix}_pop_cosine"] = (
            cosine_sum / cosine_count if cosine_count > 0 else 0.0
        )

        # --- Fano-stratified Pearson r (KOSMOS recommendation) ---
        # Reports per-neuron r for sub-Poisson (FF<1), near-Poisson
        # (1≤FF≤1.5), and super-Poisson (FF>1.5) neuron populations.
        if self.fano_factors is not None and per_ch_r is not None:
            import numpy as np
            per_ch_r_np = per_ch_r.cpu().numpy()
            ff = self.fano_factors
            n_ch = min(len(ff), len(per_ch_r_np))
            ff = ff[:n_ch]
            r_np = per_ch_r_np[:n_ch]

            # Sub-Poisson: FF < 1 (hardest to predict, most regular)
            sub_mask = ff < 1.0
            if sub_mask.sum() > 0:
                metrics[f"{prefix}_r_sub_poisson"] = float(
                    np.mean(r_np[sub_mask])
                )
                metrics[f"{prefix}_n_sub_poisson"] = int(sub_mask.sum())

            # Near-Poisson: 1.0 ≤ FF ≤ 1.5
            near_mask = (ff >= 1.0) & (ff <= 1.5)
            if near_mask.sum() > 0:
                metrics[f"{prefix}_r_near_poisson"] = float(
                    np.mean(r_np[near_mask])
                )
                metrics[f"{prefix}_n_near_poisson"] = int(near_mask.sum())

            # Super-Poisson: FF > 1.5 (most predictable, bursty)
            super_mask = ff > 1.5
            if super_mask.sum() > 0:
                metrics[f"{prefix}_r_super_poisson"] = float(
                    np.mean(r_np[super_mask])
                )
                metrics[f"{prefix}_n_super_poisson"] = int(super_mask.sum())

        # --- Per-region Pearson r (if region_map is available) ---
        # Aggregates per-channel r values by brain region, giving
        # metrics like val_pearson_r_VISp, val_pearson_r_CA1, etc.
        if self.region_map is not None and per_ch_r is not None:
            # Group channel indices by region
            from collections import defaultdict
            region_channels: Dict[str, list] = defaultdict(list)
            for ch_idx, region in self.region_map.items():
                if ch_idx < len(per_ch_r):
                    region_channels[region].append(ch_idx)

            per_ch_r_cpu = per_ch_r.cpu()
            ch_weight_cpu = ch_weight.cpu()

            for region, ch_indices in sorted(region_channels.items()):
                if not ch_indices:
                    continue
                # Weighted mean r for this region
                idx_t = torch.tensor(ch_indices, dtype=torch.long)
                region_r = per_ch_r_cpu[idx_t]
                region_w = ch_weight_cpu[idx_t]
                w_sum = region_w.sum().clamp(min=1e-8)
                region_r_val = float((region_r * region_w).sum() / w_sum)
                # Sanitize region name for metric key (replace / with _)
                safe_name = region.replace("/", "_").replace(" ", "_")
                metrics[f"{prefix}_pearson_r_{safe_name}"] = region_r_val

        # --- Ceiling efficiency (KOSMOS recommendation #1) ---
        # Reports r / r_empirical_ceiling for neurons above noise floor.
        if self.empirical_ceilings is not None and per_ch_r is not None:
            import numpy as np
            per_ch_r_np = per_ch_r.cpu().numpy()
            n_ch = min(len(self.empirical_ceilings), len(per_ch_r_np))
            mean_eff, _ = ceiling_efficiency(
                per_ch_r_np[:n_ch],
                self.empirical_ceilings[:n_ch],
                min_ceiling=0.05,
            )
            metrics[f"{prefix}_ceiling_efficiency"] = mean_eff

        return metrics

    def _validate(self) -> Dict[str, float]:
        """
        Run validation on the val_loader.

        Delegates to :meth:`evaluate` with ``prefix="val"``.

        Returns:
            Dict with val_loss, val_poisson_nll, val_pearson_r, val_mae, val_mse.
        """
        return self.evaluate(self.val_loader, prefix="val")

    @torch.no_grad()
    def evaluate_population(
        self,
        loader: DataLoader,
        max_samples: int = 100_000,
    ) -> Dict[str, object]:
        """
        Compute population-level metrics (co-BPS, calibration).

        Runs a second inference pass to collect full (T, M) prediction
        and ground truth arrays, then delegates to
        ``src.eval.population_metrics.compute_population_metrics()``.

        This is more expensive than streaming eval because it stores
        all predictions in memory. Use max_samples to cap memory usage
        on very large datasets.

        Called at end of training (not per-epoch) and during post-hoc
        evaluation. Results are suitable for W&B logging and S3 upload.

        Args:
            loader: DataLoader to evaluate on.
            max_samples: Maximum number of (time-bin) samples to collect.
                Prevents OOM on very large multi-session datasets.

        Returns:
            Dict with co_bps, calibration_error, calibration_slope, etc.
            All values are JSON-serializable for W&B and S3.
        """
        from src.eval.population_metrics import compute_population_metrics

        self.model.eval()

        # Collect predictions and ground truth into lists
        all_gt = []
        all_pred = []
        n_collected = 0

        for batch in loader:
            # Unpack batch: (x, y), (x, y, mask), or (x, y, mask, covariates)
            if len(batch) == 4:
                x, y, batch_mask, covariates = batch
                batch_mask = batch_mask.to(self.device)
                covariates = covariates.to(self.device)
            elif len(batch) == 3:
                x, y, batch_mask = batch
                batch_mask = batch_mask.to(self.device)
                covariates = None
            else:
                x, y = batch
                batch_mask = None
                covariates = None

            x = x.to(self.device)
            y = y.to(self.device)

            # Read session_id from cycling loader
            session_id = getattr(loader, 'current_session_id', None)

            # Forward pass
            with torch.autocast(
                self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                y_hat = self.model(
                    x, covariates=covariates, session_id=session_id,
                )
            if isinstance(y_hat, tuple):
                y_hat = y_hat[0]
            y_hat = y_hat.float()

            # Align shapes for session-specific heads
            out_dim = y_hat.shape[-1]
            if y.shape[-1] != out_dim:
                y = y[:, :out_dim]
                if batch_mask is not None:
                    batch_mask = batch_mask[:, :out_dim]

            # Move to CPU numpy for population metrics
            gt_np = y.cpu().numpy()
            pred_np = y_hat.cpu().numpy()

            # Apply mask: zero out padded channels
            if batch_mask is not None:
                mask_np = batch_mask.cpu().numpy()
                gt_np = gt_np * mask_np
                pred_np = pred_np * mask_np

            all_gt.append(gt_np)
            all_pred.append(pred_np)
            n_collected += gt_np.shape[0]

            if n_collected >= max_samples:
                break

        if not all_gt:
            logger.warning("evaluate_population: no samples collected")
            return {"co_bps": 0.0, "calibration_error": 0.0}

        # Concatenate into (T, M) arrays
        gt_arr = np.concatenate(all_gt, axis=0)
        pred_arr = np.concatenate(all_pred, axis=0)

        logger.info(
            "evaluate_population: collected %d samples × %d channels",
            gt_arr.shape[0], gt_arr.shape[1],
        )

        # Compute population metrics (co-BPS + calibration + PSTH R²)
        # Extract trial data for PSTH R² if behavior_data is available
        trial_ids = None
        condition_labels = None
        if self.behavior_data is not None:
            try:
                trial_idx = self.behavior_data.get("trial_index")
                resp = self.behavior_data.get("response_choice")
                left_c = self.behavior_data.get("left_contrast")
                right_c = self.behavior_data.get("right_contrast")

                if trial_idx is not None and resp is not None:
                    # Map from full-session indices to our collected subset
                    n_collected = gt_arr.shape[0]
                    trial_ids = trial_idx[:n_collected].astype(int)

                    # Build condition labels: combine stimulus + response
                    # into a single condition per trial
                    unique_trials = np.unique(trial_ids)
                    unique_trials = unique_trials[unique_trials >= 0]
                    n_trials = int(unique_trials.max()) + 1 if len(unique_trials) > 0 else 0
                    condition_labels = np.zeros(n_trials, dtype=int)

                    for tid in unique_trials:
                        mask = trial_ids == tid
                        # Use response + contrast bins as condition
                        r = int(resp[mask][0])  # -1, 0, or 1
                        lc = float(left_c[mask][0]) if left_c is not None else 0
                        rc = float(right_c[mask][0]) if right_c is not None else 0
                        # Hash to a condition label
                        condition_labels[tid] = hash((r, round(lc, 2), round(rc, 2))) % 10000

                    logger.info(
                        "PSTH R²: %d trials, %d unique conditions",
                        len(unique_trials),
                        len(np.unique(condition_labels[unique_trials])),
                    )
            except Exception as e:
                logger.warning("Could not extract trial data for PSTH R²: %s", e)

        results = compute_population_metrics(
            gt_arr, pred_arr,
            trial_ids=trial_ids,
            condition_labels=condition_labels,
        )

        # Filter out non-scalar values for W&B compatibility
        # (bin_pred_means, bin_obs_means are lists — keep for S3 but
        # mark separately for W&B)
        scalar_results = {
            k: v for k, v in results.items()
            if isinstance(v, (int, float, type(None)))
        }

        logger.info(
            "Population metrics: co_bps=%.4f, cal_error=%.4f, cal_slope=%.3f",
            scalar_results.get("co_bps", 0.0),
            scalar_results.get("calibration_error", 0.0),
            scalar_results.get("calibration_slope", 0.0),
        )

        return results

    def _save_checkpoint(self, path: Path, epoch: int, is_best: bool) -> None:
        """
        Save a model checkpoint.

        Args:
            path: File path for the checkpoint.
            epoch: Current epoch number.
            is_best: Whether this is the best model so far.
        """
        # Strip _orig_mod. prefix from torch.compile() so checkpoints
        # are loadable by non-compiled models (and vice versa).
        raw_sd = self.model.state_dict()
        clean_sd = {
            k.replace("_orig_mod.", ""): v for k, v in raw_sd.items()
        }
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": clean_sd,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
            "history": self.history,
            "global_step": self.global_step,  # For step-based warmup resume
        }
        torch.save(checkpoint, path)
        logger.info(
            "Saved %s checkpoint at epoch %d (step %d) to %s",
            "best" if is_best else "final", epoch, self.global_step, path,
        )

    def load_checkpoint(self, path: Union[str, Path]) -> int:
        """
        Load a model checkpoint.

        Args:
            path: Path to the checkpoint file.

        Returns:
            The epoch number from the checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        # Strip _orig_mod. prefix for backward compat with compiled checkpoints
        sd = {
            k.replace("_orig_mod.", ""): v
            for k, v in checkpoint["model_state_dict"].items()
        }
        self.model.load_state_dict(sd)
        # Optimizer/scheduler state might not match if the checkpoint was
        # trained with a different loss (e.g., CMP has extra learnable params).
        # Gracefully skip on mismatch — eval-only doesn't need these.
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except (ValueError, KeyError) as e:
            logger.warning(
                "Skipped optimizer/scheduler restore (param mismatch — "
                "safe for eval-only): %s", e,
            )
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint.get("history", self.history)
        self.epochs_without_improvement = checkpoint.get(
            "epochs_without_improvement", 0,
        )
        # Restore step counter for step-based warmup resume
        self.global_step = checkpoint.get("global_step", self.warmup_steps)
        epoch = checkpoint["epoch"]
        self.start_epoch = epoch + 1  # resume from next epoch
        logger.info("Loaded checkpoint from epoch %d (step %d)", epoch, self.global_step)
        return epoch

    def overfit_one_batch(
        self,
        n_iters: int = 200,
        log_every: int = 10,
    ) -> Dict[str, Any]:
        """
        Overfit on a single batch for n_iters steps (sanity check).

        Grabs the **first batch** from train_loader and runs n_iters
        forward/backward steps on it.  If the model and loss are wired
        correctly, the loss should drop to near zero.

        This is a diagnostic utility — it does NOT modify the main
        training loop, checkpoints, or history.

        Args:
            n_iters: Number of gradient update steps.
            log_every: Log loss every N steps.

        Returns:
            Dict with:
                - "losses": list of loss values at each step
                - "initial_loss": first loss value
                - "final_loss": last loss value
                - "loss_ratio": final_loss / initial_loss
                - "converged": True if final_loss < 0.2 * initial_loss
                  (5x reduction — strict but achievable on real spike data)
        """
        # Grab the first batch from the train loader
        batch = next(iter(self.train_loader))

        # Unpack batch: (x, y), (x, y, mask), or (x, y, mask, covariates)
        if len(batch) == 4:
            x, y, mask, covariates = batch
            mask = mask.to(self.device)
            covariates = covariates.to(self.device)
        elif len(batch) == 3:
            x, y, mask = batch
            mask = mask.to(self.device)
            covariates = None
        else:
            x, y = batch
            mask = None
            covariates = None

        # Move to device once (reused every iteration)
        x = x.to(self.device)
        y = y.to(self.device)

        # Read session_id from cycling loader (None for standard loaders)
        session_id = getattr(self.train_loader, 'current_session_id', None)

        logger.info(
            "Overfit sanity check: %d iters on batch of %d samples "
            "(x shape=%s, y shape=%s)",
            n_iters, x.shape[0], tuple(x.shape), tuple(y.shape),
        )

        self.model.train()
        losses: List[float] = []

        for step in range(1, n_iters + 1):
            # Forward pass under autocast (matches train loop behavior)
            with torch.autocast(
                self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                y_hat = self.model(
                    x, covariates=covariates, session_id=session_id,
                )
                # Unpack tuple returns (e.g. SNN returns (rates, spikes))
                if isinstance(y_hat, tuple):
                    y_hat = y_hat[0]
                loss = self._compute_loss(y_hat, y, mask=mask)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )

            self.optimizer.step()

            loss_val = loss.item()
            losses.append(loss_val)

            if step % log_every == 0 or step == 1:
                logger.info(
                    "  [overfit] step %3d/%d  loss=%.6f", step, n_iters, loss_val,
                )

        initial_loss = losses[0]
        final_loss = losses[-1]
        # Avoid division by zero if initial loss is 0 (unlikely but safe)
        loss_ratio = final_loss / max(initial_loss, 1e-12)
        # 5x reduction = success (strict but achievable on real spike data,
        # where Poisson NLL has an irreducible noise floor)
        converged = loss_ratio < 0.2

        logger.info(
            "Overfit sanity check complete: initial=%.6f, final=%.6f, "
            "ratio=%.4f, converged=%s",
            initial_loss, final_loss, loss_ratio, converged,
        )

        return {
            "losses": losses,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_ratio": loss_ratio,
            "converged": converged,
        }

    def train(self) -> Dict[str, List[float]]:
        """
        Run the full training loop.

        Returns:
            Training history dict with per-epoch metrics.
        """
        logger.info("Starting training: %d epochs", self.epochs)
        start_time = time.time()

        for epoch in range(self.start_epoch, self.epochs + 1):
            # Train one epoch
            train_loss = self._train_one_epoch()

            # Decide whether to validate this epoch:
            # Always validate on the last epoch and on val_every_n intervals
            is_last_epoch = (epoch == self.epochs)
            should_validate = (
                is_last_epoch
                or (self.val_every_n <= 1)
                or (epoch % self.val_every_n == 0)
            )

            if should_validate:
                # Validate
                val_metrics = self._validate()
                val_loss = val_metrics["val_loss"]
            else:
                # Skip validation — use previous val_loss for logging
                val_metrics = {
                    "val_loss": self.history["val_loss"][-1] if self.history["val_loss"] else float("inf"),
                    "val_pearson_r": self.history["val_pearson_r"][-1] if self.history["val_pearson_r"] else 0.0,
                    "val_mae": self.history["val_mae"][-1] if self.history["val_mae"] else 0.0,
                }
                val_loss = val_metrics["val_loss"]

            # Get current LR
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Step the LR scheduler (only after warmup completes)
            if self.scheduler is not None and self.global_step >= self.warmup_steps:
                self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["learning_rate"].append(current_lr)

            # Record all validation metrics dynamically
            for k, v in val_metrics.items():
                if k not in self.history:
                    self.history[k] = []
                self.history[k].append(v)

            # Log progress
            skip_tag = "" if should_validate else " [val skipped]"
            logger.info(
                "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | "
                "val_r=%.4f | val_MAE=%.4f | lr=%.2e%s",
                epoch, self.epochs, train_loss, val_loss,
                val_metrics.get("val_pearson_r", 0.0),
                val_metrics.get("val_mae", 0.0),
                current_lr, skip_tag,
            )

            # --- W&B per-epoch logging (multi-metric, per KOSMOS) ---
            if self.use_wandb:
                wandb_metrics = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "learning_rate": current_lr,
                }
                # Only log best_val_loss once we have a real value
                if self.best_val_loss < float("inf"):
                    wandb_metrics["best_val_loss"] = self.best_val_loss
                    wandb_metrics["best_val_pearson_r"] = max(
                        self.history.get("val_pearson_r", [0.0])
                    )
                # Only log validation metrics when we actually validated
                if should_validate and val_loss < float("inf"):
                    # Core metrics
                    wandb_metrics.update({
                        "val_loss": val_loss,
                        "val_pearson_r": val_metrics.get("val_pearson_r", 0.0),
                        "val_mae": val_metrics.get("val_mae", 0.0),
                        "val_mse": val_metrics.get("val_mse", 0.0),
                    })
                    # Fano-stratified metrics (KOSMOS)
                    for key in (
                        "val_r_sub_poisson", "val_r_near_poisson",
                        "val_r_super_poisson",
                    ):
                        if key in val_metrics:
                            wandb_metrics[key] = val_metrics[key]
                    # Per-region metrics
                    for key, val in val_metrics.items():
                        if key.startswith("val_pearson_r_"):
                            wandb_metrics[key] = val
                    # Ceiling efficiency (KOSMOS #1)
                    if "val_ceiling_efficiency" in val_metrics:
                        wandb_metrics["val_ceiling_efficiency"] = (
                            val_metrics["val_ceiling_efficiency"]
                        )
                    # Population-level metrics (system dynamics)
                    for key in (
                        "val_pop_rate_r", "val_spatial_r",
                        "val_pop_cosine",
                    ):
                        if key in val_metrics:
                            wandb_metrics[key] = val_metrics[key]
                wandb.log(wandb_metrics, step=epoch)

            # Check for improvement (only when we actually validated)
            if should_validate and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0

                # Save best model checkpoint
                if self.exp_dir is not None:
                    best_path = self.exp_dir / "best_model.pt"
                    self._save_checkpoint(best_path, epoch, is_best=True)
                    # Fire callback (e.g., S3 upload on NRP)
                    if self.checkpoint_callback is not None:
                        try:
                            self.checkpoint_callback(best_path, epoch)
                        except Exception as e:
                            logger.warning(
                                "Checkpoint callback failed at epoch %d: %s",
                                epoch, e,
                            )
            elif should_validate:
                self.epochs_without_improvement += 1

            # Fire metrics callback after each validation pass
            # (e.g., NRP uploads metrics.json to S3 incrementally)
            if should_validate and self.metrics_callback is not None:
                try:
                    self.metrics_callback(epoch, self.history)
                except Exception as e:
                    logger.warning(
                        "Metrics callback failed at epoch %d: %s",
                        epoch, e,
                    )

            # Early stopping (only counts validated epochs)
            if self.epochs_without_improvement >= self.patience:
                logger.info(
                    "Early stopping at epoch %d (no improvement for %d epochs)",
                    epoch, self.patience,
                )
                break

        # Save final model
        if self.exp_dir is not None:
            final_path = self.exp_dir / "final_model.pt"
            self._save_checkpoint(final_path, epoch, is_best=False)

        # --- End-of-training population metrics ---
        # Run a second inference pass on validation data to compute
        # population-level metrics (co-BPS, calibration). These are
        # too expensive for per-epoch but valuable for final eval.
        try:
            pop_metrics = self.evaluate_population(self.val_loader)

            # Store as a nested dict (not flat keys) to avoid
            # breaking per-epoch history length assertions.
            pop_scalars = {
                k: v for k, v in pop_metrics.items()
                if isinstance(v, (int, float, type(None)))
            }
            self.history["population_metrics"] = pop_scalars

            # Log to W&B as summary metrics
            if self.use_wandb:
                wandb_pop = {
                    f"pop_{k}": v for k, v in pop_scalars.items()
                    if v is not None
                }
                wandb.run.summary.update(wandb_pop)
                logger.info("Population metrics logged to W&B summary")
        except Exception as e:
            logger.warning("Population metrics failed (non-fatal): %s", e)

        elapsed = time.time() - start_time
        logger.info(
            "Training complete in %.1fs. Best val_loss=%.4f at epoch %d",
            elapsed, self.best_val_loss,
            self.history["val_loss"].index(self.best_val_loss) + 1,
        )

        # Finalize W&B run (deferred when NRP wrapper handles finalization,
        # so nrp_train.py can log final summary metrics before closing)
        if self.use_wandb and not os.environ.get("NRP_DEFER_WANDB_FINISH"):
            wandb.finish()
            logger.info("W&B run finalized")

        return self.history
