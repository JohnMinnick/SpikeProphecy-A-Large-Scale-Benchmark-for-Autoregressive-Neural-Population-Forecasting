"""
Train a multi-head Student SNN via distillation from a Mamba teacher.

Extends the standard distillation (dynamics-only) with auxiliary heads
that decode behavioral variables:
    - Stimulus head: predicts left/right visual contrast (regression)
    - Response head: predicts animal's response choice (classification)

The auxiliary heads share the spiking backbone with the dynamics head.
Their losses are trial-masked (only computed during active trials).

Usage:
    # NRP container:
    python scripts/train_distill_multi_head.py \\
        --teacher-s3-slug 2026-03-17_mamba-baseline-v1 \\
        --teacher-config configs/teacher/nrp_teacher_mamba.yaml \\
        --data-config configs/data/steinmetz_multi_nrp_50ms_no_cov.yaml \\
        --student-config configs/student/distill_mamba_multi_head.yaml \\
        --slug distill-mamba-multi-head-v1
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.behavior_loader import extract_trial_stimuli
from src.data.multi_session_loader import (
    preprocess_and_cache,
    create_dataloaders,
    MaskedSpikeCountDataset,
    pad_to_channels,
    build_channel_mask,
)
from src.distill.multi_head_loss import MultiHeadDistillationLoss
from src.distill.multi_head_trainer import MultiHeadDistillTrainer
from src.models.student import StudentSNN
from src.models.teacher import create_teacher_model
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.experiment import create_experiment
from src.utils.seed import seed_everything

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_distill_multi_head")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for multi-head SNN distillation."""
    parser = argparse.ArgumentParser(
        description="Train multi-head Student SNN via distillation.",
    )
    parser.add_argument(
        "--teacher-config", type=str, required=True,
        help="Path to teacher model config YAML.",
    )
    parser.add_argument(
        "--data-config", type=str, required=True,
        help="Path to data config YAML.",
    )
    parser.add_argument(
        "--student-config", type=str, required=True,
        help="Path to student model config YAML.",
    )
    parser.add_argument(
        "--teacher-checkpoint", type=str, default="",
        help="Local path to pretrained teacher checkpoint (.pt).",
    )
    parser.add_argument(
        "--teacher-s3-slug", type=str, default="",
        help="S3 experiment slug to download teacher checkpoint from.",
    )
    parser.add_argument(
        "--slug", type=str, default="distill-multi-head",
        help="Experiment slug for folder naming.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override max training epochs.",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate.",
    )
    parser.add_argument(
        "--stimulus-weight", type=float, default=None,
        help="Override stimulus loss weight (λ_stim). Default from YAML.",
    )
    parser.add_argument(
        "--response-weight", type=float, default=None,
        help="Override response loss weight (λ_resp). Default from YAML.",
    )
    parser.add_argument(
        "--distill-weight", type=float, default=None,
        help="Override KL distillation weight (β). Set 0.0 to disable KL. Default from YAML.",
    )
    parser.add_argument(
        "--mimetic-init", action="store_true", default=False,
        help="Enable mimetic initialization: copy teacher input/output projection "
             "weights into student and calibrate SNN thresholds to teacher activations.",
    )
    parser.add_argument(
        "--hidden-align-weight", type=float, default=None,
        help="Override hidden-state alignment weight (λ_align). Default from YAML.",
    )
    parser.add_argument(
        "--student-checkpoint", type=str, default="",
        help="Local path to a pretrained student checkpoint (.pt) for warm-start. "
             "Loads weights before distillation training begins.",
    )
    parser.add_argument(
        "--student-s3-slug", type=str, default="",
        help="S3 experiment slug to download pretrained student checkpoint from "
             "for warm-start distillation (e.g., 'snn-standalone-v12b').",
    )
    return parser.parse_args()


class BehaviorAugmentedDataset(Dataset):
    """
    Wraps a MaskedSpikeCountDataset and appends behavioral targets.

    For sample index i, the target time bin in the full recording is
    split_start + i + history_bins. This dataset looks up the behavioral
    variables at that bin index from the pre-extracted per-session arrays.

    This embedding happens BEFORE DataLoader shuffling, so behavioral
    targets stay correctly aligned even after shuffle.
    """

    def __init__(self, base_dataset, behavior_arrays, split_start, history_bins):
        """
        Args:
            base_dataset: MaskedSpikeCountDataset returning (x, y).
            behavior_arrays: Dict with keys left_contrast, right_contrast,
                response_choice, trial_active — all shape (T_full,) for the
                full session recording.
            split_start: Start index of this split in the full recording.
            history_bins: Number of history bins (window size).
        """
        self.base = base_dataset
        self.behavior = behavior_arrays
        self.split_start = split_start
        self.history_bins = history_bins

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        """Return (x, y, behavior_dict, channel_mask) for sample idx."""
        base_out = self.base[idx]
        x, y = base_out[0], base_out[1]
        # Preserve channel mask from base dataset — 1 = real, 0 = padding
        channel_mask = base_out[2]  # (M_max,)

        # Map sample index → bin in full recording
        # Sample idx targets the bin at split_start + idx + history_bins
        target_bin = self.split_start + idx + self.history_bins

        # Look up behavioral targets at the target bin
        if (
            self.behavior is not None
            and 0 <= target_bin < len(self.behavior["trial_active"])
        ):
            beh = {
                "left_contrast": torch.tensor(
                    self.behavior["left_contrast"][target_bin],
                    dtype=torch.float32,
                ),
                "right_contrast": torch.tensor(
                    self.behavior["right_contrast"][target_bin],
                    dtype=torch.float32,
                ),
                "response_choice": torch.tensor(
                    self.behavior["response_choice"][target_bin],
                    dtype=torch.float32,
                ),
                "trial_active": torch.tensor(
                    self.behavior["trial_active"][target_bin],
                    dtype=torch.float32,
                ),
                "behavior_train_mask": torch.tensor(
                    self.behavior.get(
                        "behavior_train_mask",
                        self.behavior["trial_active"],
                    )[target_bin],
                    dtype=torch.float32,
                ),
            }
        else:
            # Outside trial range — zero behavior (non-trial bin)
            beh = {
                "left_contrast": torch.tensor(0.0),
                "right_contrast": torch.tensor(0.0),
                "response_choice": torch.tensor(0.0),
                "trial_active": torch.tensor(0.0),
                "behavior_train_mask": torch.tensor(0.0),
            }

        return x, y, beh, channel_mask


def _collate_with_behavior(batch):
    """
    Custom collate function for BehaviorAugmentedDataset.

    Stacks x, y, and channel_mask tensors, and collates behavior
    dicts into batched tensors.
    """
    xs, ys, behs, masks = zip(*batch)
    x_batch = torch.stack(xs, dim=0)
    y_batch = torch.stack(ys, dim=0)
    mask_batch = torch.stack(masks, dim=0)  # (B, M_max)

    beh_batch = {
        key: torch.stack([b[key] for b in behs], dim=0)
        for key in behs[0].keys()
    }

    return x_batch, y_batch, beh_batch, mask_batch


class MultiHeadDistillLoaderWrapper:
    """
    Replicates SessionCyclingLoader iteration but injects behavioral data
    per-sample and runs the frozen teacher per-batch.

    Yields: (x, y, teacher_rates, behavior_dict, teacher_hidden)

    teacher_hidden is the teacher's last-layer hidden state (batch, T, d_model)
    BEFORE readout, used for hidden-state alignment distillation.

    The key insight: by embedding behavioral targets at the dataset level
    (via BehaviorAugmentedDataset), the data stays aligned even after
    DataLoader shuffling.
    """

    def __init__(self, base_loader, teacher_model, dev, out_channels,
                 session_behaviors):
        """
        Args:
            base_loader: SessionCyclingLoader (used for metadata, not iterated).
            teacher_model: Frozen teacher model on device.
            dev: torch.device for teacher inference.
            out_channels: M_max (number of output neurons).
            session_behaviors: Dict mapping session_id -> behavior arrays.
        """
        self.base_loader = base_loader
        self.teacher_model = teacher_model
        self.dev = dev
        self.out_channels = out_channels
        self.session_behaviors = session_behaviors

    def __iter__(self):
        """
        Iterate session-by-session, building behavior-augmented datasets
        and running teacher inference per batch.
        """
        loader = self.base_loader  # SessionCyclingLoader

        # Determine session order (shuffle for train)
        import random
        session_indices = list(range(loader.metadata["num_sessions"]))
        if loader.shuffle_sessions:
            random.shuffle(session_indices)

        for sess_idx in session_indices:
            sess_info = loader.metadata["sessions"][sess_idx]
            session_id = f"session_{sess_idx:03d}"

            # 1. Load cached count matrix
            npy_path = loader.cache_dir / f"session_{sess_idx:03d}.npy"
            if not npy_path.exists():
                continue
            counts_u8 = np.load(npy_path)

            # 2. Get split slice
            split_start, split_end = loader._get_split_slice(sess_info)
            split_len = split_end - split_start
            if split_len <= loader.history_bins:
                del counts_u8
                continue

            counts_split = counts_u8[:, split_start:split_end].astype(np.int32)
            del counts_u8

            # 3. Pad to M_max
            m_i = counts_split.shape[0]
            padded = pad_to_channels(counts_split, loader.m_max)
            del counts_split

            # 4. Build mask
            mask = build_channel_mask(m_i, loader.m_max)
            session_masks = mask.reshape(1, -1)
            mask_index = np.zeros(padded.shape[1], dtype=np.int32)

            # 5. Create base spike dataset
            base_ds = MaskedSpikeCountDataset(
                spike_counts=padded,
                mask_index=mask_index,
                session_masks=session_masks,
                history_bins=loader.history_bins,
                output_channels=loader.m_max,
            )
            del padded, mask_index

            if len(base_ds) == 0:
                del base_ds
                continue

            # 6. Get behavioral data for this session
            beh_arrays = self.session_behaviors.get(session_id, None)

            # 7. Create behavior-augmented dataset
            aug_ds = BehaviorAugmentedDataset(
                base_dataset=base_ds,
                behavior_arrays=beh_arrays,
                split_start=split_start,
                history_bins=loader.history_bins,
            )

            # 8. Create DataLoader with custom collate
            import platform
            if platform.system() == "Windows":
                num_workers = 0
                pin_memory = False
            else:
                num_workers = loader.num_workers
                pin_memory = loader.pin_memory

            dl = DataLoader(
                aug_ds,
                batch_size=loader.batch_size,
                shuffle=loader.shuffle_samples,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=(loader.split == "train"),
                collate_fn=_collate_with_behavior,
            )

            # 9. Yield batches with teacher inference + hidden states
            for x, y, behavior, channel_mask in dl:
                with torch.no_grad():
                    x_dev = x.to(self.dev)

                    # Extract teacher hidden states BEFORE readout.
                    # This runs the Mamba forward manually to capture
                    # intermediate representations for alignment loss,
                    # then derives teacher rates from the same hidden states
                    # (avoids a double forward pass).
                    projected = self.teacher_model.input_proj(x_dev)
                    if hasattr(self.teacher_model, 'input_norm'):
                        projected = self.teacher_model.input_norm(projected)
                    hidden = projected
                    for block in self.teacher_model.mamba_blocks:
                        hidden = block(hidden)
                    hidden = self.teacher_model.final_norm(hidden)
                    teacher_hidden = hidden  # (batch, T, d_model)

                    # Complete the readout path to get teacher rates:
                    # attention/last-step → output_norm → output_proj → softplus
                    if self.teacher_model.attn_query is not None:
                        attn_scores = self.teacher_model.attn_query(hidden)
                        attn_weights = torch.softmax(attn_scores, dim=1)
                        context = (hidden * attn_weights).sum(dim=1)
                    else:
                        context = hidden[:, -1, :]

                    context = self.teacher_model.output_norm(context)
                    raw_output = self.teacher_model.output_proj(context)

                    # Cross-neuron coupling (if present)
                    if (hasattr(self.teacher_model, 'coupling')
                            and self.teacher_model.coupling is not None
                            and self.teacher_model.session_dims is None):
                        raw_output = self.teacher_model.coupling(raw_output)

                    teacher_rates = self.teacher_model.softplus(raw_output)

                    if self.out_channels is not None:
                        teacher_rates = teacher_rates[:, :self.out_channels]
                        y = y[:, :self.out_channels]
                        # Slice mask to match output channels
                        channel_mask = channel_mask[:, :self.out_channels]

                yield (
                    x.cpu(), y.cpu(), teacher_rates.cpu(),
                    behavior, teacher_hidden.cpu(),
                    channel_mask.cpu(),
                )

            # 10. Free memory
            del dl, aug_ds, base_ds
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __len__(self):
        return len(self.base_loader)


def load_session_behaviors(
    data_config: dict,
    multi_meta: dict,
    holdout_frac: float = 0.2,
    holdout_seed: int = 42,
) -> dict:
    """
    Pre-extract behavioral data from all NWB sessions with trial holdout.

    Uses the metadata from preprocess_and_cache to get the exact bin count
    and duration per session, ensuring behavioral targets align precisely
    with the spike count bins used for training.

    For each session, randomly holds out ~holdout_frac of trials for
    evaluation. Adds 'behavior_train_mask' to each session's behavior dict:
    1.0 for train-trial bins, 0.0 for held-out trials and non-trial bins.

    Args:
        data_config: Data configuration dict with NWB paths and bin settings.
        multi_meta: Metadata dict from preprocess_and_cache() with per-session
            info (num_bins, duration_s).
        holdout_frac: Fraction of trials to hold out per session (default 0.2).
        holdout_seed: Random seed for trial holdout (default 42). Must match
            at training and eval time for consistent splits.

    Returns:
        Dict mapping session_id (str) -> behavior dict with keys:
            left_contrast, right_contrast, response_choice, trial_active,
            trial_index, behavior_train_mask
    """
    # Use the glob pattern from the data config to find NWB files
    # This matches the same files that preprocess_and_cache uses
    source_glob = data_config.get("source", {}).get("glob", "data/raw/*.nwb")
    nwb_glob_path = Path(source_glob)
    data_dir = nwb_glob_path.parent
    glob_pattern = nwb_glob_path.name

    bin_width_ms = data_config.get("bin_width_ms", 50.0)
    bin_width_s = bin_width_ms / 1000.0

    # Find all NWB files matching the config glob
    nwb_files = sorted(data_dir.glob(glob_pattern))
    if not nwb_files:
        logger.warning("No NWB files found matching %s", source_glob)
        return {}

    # Get per-session metadata from preprocess_and_cache
    sessions_meta = multi_meta.get("sessions", [])

    session_behaviors = {}
    total_train_trials = 0
    total_eval_trials = 0

    for si, nwb_path in enumerate(nwb_files):
        session_id = f"session_{si:03d}"

        try:
            # Use the exact num_bins from preprocessing metadata
            # This is the ground truth for how many bins the spike data has
            if si < len(sessions_meta):
                n_bins = sessions_meta[si]["num_bins"]
                duration_s = sessions_meta[si].get(
                    "duration_s", n_bins * bin_width_s,
                )
            else:
                logger.warning(
                    "Session %s not in preprocessing metadata, skipping",
                    session_id,
                )
                continue

            # Compute bin edges matching bin_spike_trains exactly:
            # Bins are [0*bw, 1*bw, 2*bw, ..., n_bins*bw] in seconds
            bin_edges = np.arange(n_bins + 1) * bin_width_s

            # Extract trial stimuli aligned to these bins
            behavior = extract_trial_stimuli(str(nwb_path), bin_edges)

            # ---------------------------------------------------------------
            # Trial-level holdout: randomly hold out ~holdout_frac of trials
            # ---------------------------------------------------------------
            trial_indices = behavior["trial_index"]
            unique_trials = np.unique(trial_indices[trial_indices >= 0])
            n_trials = len(unique_trials)

            # Deterministic per-session RNG (session index as seed offset)
            rng = np.random.RandomState(holdout_seed + si)
            n_eval = max(1, int(n_trials * holdout_frac))
            eval_trials = set(
                rng.choice(unique_trials, size=n_eval, replace=False).tolist()
            )
            train_trials = set(unique_trials.tolist()) - eval_trials

            # Build behavior_train_mask: 1.0 for train trials, 0.0 otherwise
            behavior_train_mask = np.zeros(n_bins, dtype=np.float32)
            for t_idx in train_trials:
                behavior_train_mask[trial_indices == t_idx] = 1.0

            behavior["behavior_train_mask"] = behavior_train_mask

            total_train_trials += len(train_trials)
            total_eval_trials += len(eval_trials)

            # Convert to float32 arrays for efficiency
            session_behaviors[session_id] = {
                k: v.astype(np.float32) if v.dtype != np.float32 else v
                for k, v in behavior.items()
            }

            n_active = int(behavior["trial_active"].sum())
            n_train_bins = int(behavior_train_mask.sum())
            n_eval_bins = n_active - n_train_bins
            logger.info(
                "Session %s: %d bins, %d trial-active "
                "(%d train / %d eval trials, %d/%d bins), "
                "duration=%.1fs",
                session_id, n_bins, n_active,
                len(train_trials), len(eval_trials),
                n_train_bins, n_eval_bins, duration_s,
            )

        except Exception as e:
            logger.warning("Failed to load behavior for %s: %s", session_id, e)

    logger.info(
        "Loaded behavioral data for %d/%d sessions "
        "(%d train trials, %d eval trials, holdout=%.0f%%)",
        len(session_behaviors), len(nwb_files),
        total_train_trials, total_eval_trials,
        holdout_frac * 100,
    )
    return session_behaviors


def mimetic_init_from_teacher(
    student: "StudentSNN",
    teacher: nn.Module,
    train_loader,
    device: torch.device,
    n_calibration_batches: int = 10,
) -> None:
    """
    Mimetic initialization: transfer teacher projection weights to student
    and calibrate SNN thresholds to match teacher activation magnitudes.

    This gives the SNN a 'head start' by:
    1. Copying teacher's input_proj weights → student's input_proj
    2. Copying teacher's output_proj weights → student's output_proj
    3. Running calibration batches through the teacher to measure mean
       hidden activation magnitudes, then setting SNN thresholds to
       match (prevents dead/saturated neurons).

    Args:
        student: The student SNN model (modified in-place).
        teacher: The frozen teacher Mamba model.
        train_loader: Training data loader for calibration batches.
        device: Torch device.
        n_calibration_batches: Number of batches to use for threshold calibration.
    """
    import copy

    # ------------------------------------------------------------------
    # Step 1: Copy input projection weights (M → hidden_size)
    # ------------------------------------------------------------------
    # Teacher: self.input_proj = nn.Linear(M, d_model=256)
    # Student: self.input_proj = nn.Linear(M, hidden_size=256)
    if (hasattr(teacher, 'input_proj') and hasattr(student, 'input_proj')
            and teacher.input_proj.weight.shape == student.input_proj.weight.shape):
        student.input_proj.weight.data.copy_(teacher.input_proj.weight.data)
        student.input_proj.bias.data.copy_(teacher.input_proj.bias.data)
        logger.info(
            "Mimetic init: copied input_proj weights (%s)",
            list(teacher.input_proj.weight.shape),
        )
    else:
        logger.warning(
            "Mimetic init: input_proj shape mismatch, skipping. "
            "Teacher: %s, Student: %s",
            getattr(teacher, 'input_proj', None),
            getattr(student, 'input_proj', None),
        )

    # ------------------------------------------------------------------
    # Step 2: Copy output projection weights (hidden_size → M)
    # ------------------------------------------------------------------
    # Teacher: self.output_proj = nn.Linear(d_model=256, M)
    # Student: self.output_proj = nn.Linear(hidden_size=256, M)
    if (hasattr(teacher, 'output_proj') and hasattr(student, 'output_proj')
            and teacher.output_proj is not None
            and teacher.output_proj.weight.shape == student.output_proj.weight.shape):
        student.output_proj.weight.data.copy_(teacher.output_proj.weight.data)
        student.output_proj.bias.data.copy_(teacher.output_proj.bias.data)
        logger.info(
            "Mimetic init: copied output_proj weights (%s)",
            list(teacher.output_proj.weight.shape),
        )
    else:
        logger.warning(
            "Mimetic init: output_proj shape mismatch. "
            "Teacher: %s, Student: %s",
            list(teacher.output_proj.weight.shape),
            list(student.output_proj.weight.shape),
        )
        # Handle TI-LIF dimension mismatch: teacher (M, 256) → student (M, 512)
        # Mirror teacher weights into excitatory/inhibitory halves
        if (hasattr(teacher, 'output_proj') and hasattr(student, 'output_proj')
                and teacher.output_proj is not None
                and student.output_proj.weight.shape[1] == 2 * teacher.output_proj.weight.shape[1]):
            half = teacher.output_proj.weight.shape[1]  # 256
            # Excitatory half: direct copy of teacher weights
            student.output_proj.weight.data[:, :half] = teacher.output_proj.weight.data
            # Inhibitory half: negated mirror (inhibitory channels)
            student.output_proj.weight.data[:, half:] = -teacher.output_proj.weight.data
            # Bias: direct copy (shared across both halves)
            student.output_proj.bias.data.copy_(teacher.output_proj.bias.data)
            logger.info(
                "Mimetic init: TI-LIF mirrored output_proj (%s → %s)",
                list(teacher.output_proj.weight.shape),
                list(student.output_proj.weight.shape),
            )

    # ------------------------------------------------------------------
    # Step 3: Threshold calibration via teacher activation statistics
    # ------------------------------------------------------------------
    # Run calibration batches through teacher's input_proj to measure
    # the mean activation magnitude. Set SNN thresholds so that a
    # spiking neuron receiving these activations fires at a reasonable
    # rate (~50% of timesteps), not constantly or never.
    logger.info(
        "Mimetic init: calibrating SNN thresholds from %d batches...",
        n_calibration_batches,
    )

    activation_magnitudes = []
    batch_count = 0

    with torch.no_grad():
        for batch_data in train_loader:
            if batch_count >= n_calibration_batches:
                break

            # Handle different loader output formats
            if len(batch_data) == 5:
                x, y, teacher_rates, behavior, teacher_hidden = batch_data
            elif len(batch_data) == 4:
                x, y, teacher_rates, behavior = batch_data
            elif len(batch_data) == 3:
                x, y, behavior = batch_data
            else:
                x, y = batch_data[:2]

            x_dev = x.to(device)

            # Get teacher's hidden activations after input_proj
            # This is what flows into the SSM blocks / spiking layers
            projected = teacher.input_proj(x_dev)  # (batch, T, hidden)
            if hasattr(teacher, 'input_norm'):
                projected = teacher.input_norm(projected)

            # Mean absolute activation per hidden unit across batch and time
            mean_act = projected.abs().mean(dim=(0, 1))  # (hidden,)
            activation_magnitudes.append(mean_act.cpu())
            batch_count += 1

    if activation_magnitudes:
        # Average across calibration batches
        mean_activation = torch.stack(activation_magnitudes).mean(dim=0)
        global_mean = mean_activation.mean().item()
        global_std = mean_activation.std().item()

        # Set threshold to mean activation magnitude
        # This ensures ~50% firing rate under teacher-like inputs
        calibrated_threshold = max(global_mean, 0.1)  # Floor at 0.1

        logger.info(
            "Mimetic init: teacher activation stats — "
            "mean=%.4f, std=%.4f, calibrated_threshold=%.4f",
            global_mean, global_std, calibrated_threshold,
        )

        # Apply calibrated threshold to all spiking layers
        for layer_idx, spiking_layer in enumerate(student.spiking_layers):
            # TI-LIF: threshold is a property; target the underlying storage
            if hasattr(spiking_layer, 'threshold_val'):
                old_val = spiking_layer.threshold_val.mean().item()
                spiking_layer.threshold_val.fill_(calibrated_threshold)
                logger.info(
                    "  Layer %d: threshold %.4f → %.4f",
                    layer_idx, old_val, calibrated_threshold,
                )
            elif hasattr(spiking_layer, 'threshold_param'):
                old_val = nn.functional.softplus(spiking_layer.threshold_param).mean().item()
                import math
                inv_softplus = math.log(math.exp(calibrated_threshold) - 1)
                spiking_layer.threshold_param.data.fill_(inv_softplus)
                logger.info(
                    "  Layer %d: threshold %.4f → %.4f (learnable)",
                    layer_idx, old_val, calibrated_threshold,
                )
            elif hasattr(spiking_layer, 'threshold'):
                # snnTorch neurons: direct attribute
                old_threshold = spiking_layer.threshold
                if isinstance(old_threshold, (torch.Tensor, nn.Parameter)):
                    spiking_layer.threshold.data.fill_(calibrated_threshold)
                else:
                    spiking_layer.threshold = calibrated_threshold
                logger.info(
                    "  Layer %d: threshold %s → %.4f",
                    layer_idx,
                    old_threshold.item() if isinstance(old_threshold, (torch.Tensor, nn.Parameter)) else old_threshold,
                    calibrated_threshold,
                )
    else:
        logger.warning("Mimetic init: no calibration batches available")

    logger.info("Mimetic initialization complete")

    # Store calibrated threshold for threshold annealing
    if activation_magnitudes:
        student._calibrated_threshold = calibrated_threshold
        student._initial_threshold = calibrated_threshold * 0.3  # Start at 30%
        logger.info(
            "Mimetic init: threshold annealing range [%.4f, %.4f]",
            student._initial_threshold, student._calibrated_threshold,
        )


def main() -> None:
    """Main multi-head SNN distillation pipeline."""
    start_time = time.time()
    args = parse_args()

    logger.info("=" * 60)
    logger.info("MULTI-HEAD SNN DISTILLATION — SpikeProphecy (KOSMOS)")
    logger.info("=" * 60)
    logger.info("  Teacher config:     %s", args.teacher_config)
    logger.info("  Student config:     %s", args.student_config)
    logger.info("  Data config:        %s", args.data_config)
    logger.info("  Teacher checkpoint: %s", args.teacher_checkpoint or "(from S3)")
    logger.info("  Slug:               %s", args.slug)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load configs
    # ------------------------------------------------------------------
    teacher_config = load_config(args.teacher_config)
    data_config = load_config(args.data_config)
    student_config = load_config(args.student_config)

    # Apply CLI overrides
    if args.epochs is not None:
        student_config.setdefault("training", {})["epochs"] = args.epochs
    if args.lr is not None:
        student_config.setdefault("training", {})["learning_rate"] = args.lr

    # ------------------------------------------------------------------
    # 2. Download data and teacher checkpoint from S3 if NRP mode
    # ------------------------------------------------------------------
    teacher_checkpoint_path = args.teacher_checkpoint

    # Determine source type for data routing
    source_type = data_config.get("source", {}).get("type", "nwb_multi")

    if os.environ.get("AWS_ACCESS_KEY_ID"):
        try:
            # scripts/ is not a Python package (no __init__.py), so use importlib
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "nrp_train", PROJECT_ROOT / "scripts" / "nrp_train.py"
            )
            nrp_train = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(nrp_train)

            if source_type == "ibl":
                # IBL or combined data: download pre-cached .npy session
                # arrays and metadata.json from the IBL S3 prefix.
                # Auto-detect S3 prefix from config's ibl.tag field.
                ibl_tag = data_config.get("ibl", {}).get("tag", "repeated_site")
                tag_to_prefix = {
                    "repeated_site": "<anon>/spike-prophecy/inputs/ibl-repeated-site",
                    "combined": "<anon>/spike-prophecy/inputs/combined-steinmetz-ibl",
                }
                tag_to_cache = {
                    "repeated_site": "ibl_repeated_site_cache",
                    "combined": "combined_steinmetz_ibl_cache",
                }
                default_prefix = tag_to_prefix.get(
                    ibl_tag, f"<anon>/spike-prophecy/inputs/{ibl_tag}"
                )
                default_cache = tag_to_cache.get(
                    ibl_tag, f"{ibl_tag}_cache"
                )

                ibl_s3_prefix = os.environ.get("S3_IBL_PREFIX", default_prefix)
                ibl_cache_name = os.environ.get("S3_IBL_CACHE_DIR", default_cache)
                ibl_cache = PROJECT_ROOT / "data" / "processed" / ibl_cache_name
                ibl_cache.mkdir(parents=True, exist_ok=True)

                logger.info(
                    "NRP mode — downloading IBL cache from S3 prefix: %s",
                    ibl_s3_prefix,
                )

                nrp_dir = PROJECT_ROOT / "nrp"
                sys.path.insert(0, str(nrp_dir))
                from s3_utils import list_files, download_single_file

                all_keys = list_files(ibl_s3_prefix)
                ibl_keys = [
                    k for k in all_keys
                    if k.endswith(".npy") or k.endswith(".json")
                ]
                logger.info(
                    "Found %d IBL cache files in S3 under %s",
                    len(ibl_keys), ibl_s3_prefix,
                )

                for key in ibl_keys:
                    filename = os.path.basename(key)
                    local_path = str(ibl_cache / filename)
                    logger.info("Downloading %s", key)
                    download_single_file(key=key, local_path=local_path)

                # Verify metadata.json was downloaded
                meta_path = ibl_cache / "metadata.json"
                if meta_path.exists():
                    import json as _json
                    with open(meta_path) as _mf:
                        _ibl_meta = _json.load(_mf)
                    logger.info(
                        "IBL cache ready: %d sessions, M_max=%d",
                        _ibl_meta.get("num_sessions", 0),
                        _ibl_meta.get("m_max", 0),
                    )
                else:
                    logger.warning(
                        "metadata.json not found in IBL cache — "
                        "preprocess_and_cache will attempt to regenerate"
                    )
            else:
                # Steinmetz NWB data: download .nwb files from S3
                logger.info("NRP mode — downloading NWB data from S3...")
                nrp_train.download_nwb_from_s3()
        except Exception as e:
            logger.warning("Could not download from S3: %s", e)

        if args.teacher_s3_slug:
            logger.info(
                "Downloading teacher checkpoint from S3: %s",
                args.teacher_s3_slug,
            )
            try:
                from scripts.nrp_train import download_checkpoint_from_s3
                teacher_checkpoint_path = str(
                    download_checkpoint_from_s3(args.teacher_s3_slug),
                )
                logger.info("Teacher checkpoint: %s", teacher_checkpoint_path)
            except Exception as e:
                logger.error("Failed to download teacher checkpoint: %s", e)
                sys.exit(1)

    if not teacher_checkpoint_path:
        logger.error(
            "No teacher checkpoint specified. Use --teacher-checkpoint "
            "or --teacher-s3-slug.",
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Seed and device
    # ------------------------------------------------------------------
    seed = student_config.get("training", {}).get("seed", 42)
    seed_everything(seed)
    device = resolve_device()
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    # 4. Preprocess data and create base DataLoaders
    # ------------------------------------------------------------------
    # Dispatch on source.type: IBL data uses pre-cached .npy arrays
    # (downloaded from S3 in step 2), while NWB data needs NWB→cache
    # conversion via preprocess_and_cache(). Mirrors train_snn_standalone.py.
    logger.info("Preprocessing multi-session data...")

    if source_type == "ibl":
        # IBL source: use pre-cached .npy files from S3 if metadata exists
        import json as _json_preprocess

        ibl_tag = data_config.get("ibl", {}).get("tag", "repeated_site")
        tag_to_cache = {
            "repeated_site": "ibl_repeated_site_cache",
            "combined": "combined_steinmetz_ibl_cache",
        }
        ibl_cache_name = os.environ.get(
            "S3_IBL_CACHE_DIR",
            tag_to_cache.get(ibl_tag, f"{ibl_tag}_cache"),
        )
        ibl_cache_dir = Path("data/processed") / ibl_cache_name
        _meta_path = ibl_cache_dir / "metadata.json"

        if _meta_path.exists():
            with open(_meta_path, "r", encoding="utf-8") as _mf:
                multi_meta = _json_preprocess.load(_mf)
            cache_dir = ibl_cache_dir
            logger.info(
                "Found existing IBL cache metadata: %d sessions, M_max=%d "
                "— skipping re-processing",
                multi_meta["num_sessions"], multi_meta["m_max"],
            )
        else:
            from src.data.ibl_data_loader import preprocess_and_cache_ibl
            cache_dir, multi_meta = preprocess_and_cache_ibl(
                data_config, cache_dir=str(ibl_cache_dir),
            )
    else:
        # NWB source (Steinmetz): standard NWB → cache preprocessing
        cache_dir, multi_meta = preprocess_and_cache(data_config)

    m_max = multi_meta["m_max"]
    logger.info("Data ready: M_max=%d, %d sessions", m_max, multi_meta["num_sessions"])

    base_loaders = create_dataloaders(cache_dir, multi_meta, data_config)

    # ------------------------------------------------------------------
    # 5. Load behavioral data from NWB files
    # ------------------------------------------------------------------
    logger.info("Loading behavioral data from NWB files...")
    session_behaviors = load_session_behaviors(data_config, multi_meta)

    # ------------------------------------------------------------------
    # 6. Load pretrained teacher (frozen)
    # ------------------------------------------------------------------
    logger.info("Loading pretrained teacher from %s...", teacher_checkpoint_path)
    teacher = create_teacher_model(
        config=teacher_config,
        input_size=m_max,
    )
    checkpoint = torch.load(
        teacher_checkpoint_path, map_location=device, weights_only=True,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    teacher.load_state_dict(state_dict)
    teacher.to(device)
    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad_(False)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    logger.info("Teacher loaded and frozen: %d params", teacher_params)

    # ------------------------------------------------------------------
    # 7. Wrap loaders with teacher inference + behavioral data
    # ------------------------------------------------------------------
    distill_loaders = {}
    for split_name, base_loader in base_loaders.items():
        distill_loaders[split_name] = MultiHeadDistillLoaderWrapper(
            base_loader, teacher, device, m_max, session_behaviors,
        )
        logger.info(
            "Distill %s loader: wrapped with teacher + behavioral data",
            split_name,
        )

    # ------------------------------------------------------------------
    # 8. Create multi-head Student SNN
    # ------------------------------------------------------------------
    logger.info("Creating multi-head StudentSNN...")
    student_model_cfg = student_config.get("model", {})

    student = StudentSNN(
        input_size=m_max,
        hidden_size=student_model_cfg.get("hidden_size", 256),
        beta=student_model_cfg.get("beta", 0.9),
        threshold=student_model_cfg.get("threshold", 1.0),
        output_size=m_max,
        gradient_slope=student_model_cfg.get("gradient_slope", 25.0),
        learn_beta=student_model_cfg.get("learn_beta", True),
        num_layers=student_model_cfg.get("num_layers", 2),
        neuron_type=student_model_cfg.get("neuron_type", "rsynaptic"),
        alpha=student_model_cfg.get("alpha", 0.85),
        use_layer_norm=student_model_cfg.get("use_layer_norm", False),
        dropout=student_model_cfg.get("dropout", 0.0),
        learn_threshold=student_model_cfg.get("learn_threshold", False),
        readout_mode=student_model_cfg.get("readout_mode", "mean"),
        auxiliary_heads=student_model_cfg.get("auxiliary_heads", None),
        sgc_enabled=student_model_cfg.get("sgc_enabled", False),
    )
    student.to(device)

    student_params = sum(p.numel() for p in student.parameters())
    logger.info(
        "Student created: %d params (teacher: %d, ratio: %.2fx)",
        student_params, teacher_params, student_params / max(teacher_params, 1),
    )

    # ------------------------------------------------------------------
    # 8a. Warm-start: load pretrained student weights (optional)
    # ------------------------------------------------------------------
    student_ckpt_path = args.student_checkpoint
    if not student_ckpt_path and args.student_s3_slug:
        # Download from S3
        import boto3
        s3_client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("ENDPOINT",
                os.environ.get("S3_ENDPOINT",
                    "https://s3-west.nrp-nautilus.io")),
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )
        student_ckpt_path = f"/tmp/student_warmstart_{args.student_s3_slug}.pt"
        s3_key = (
            f"<anon>/spike-prophecy/outputs/{args.student_s3_slug}/best_model.pt"
        )
        logger.info(
            "Downloading student checkpoint from S3: %s", s3_key,
        )
        s3_client.download_file(
            "<lab-bucket>", s3_key, student_ckpt_path,
        )

    if student_ckpt_path:
        logger.info("=" * 60)
        logger.info("WARM-START: Loading pretrained student from %s",
                    student_ckpt_path)
        logger.info("=" * 60)
        ckpt = torch.load(student_ckpt_path, map_location=device,
                          weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        # Load with strict=False to handle minor architecture differences
        # (e.g., auxiliary heads not present in standalone checkpoint)
        missing, unexpected = student.load_state_dict(
            state_dict, strict=False,
        )
        if missing:
            logger.info("  Warm-start missing keys (expected for new heads): %s",
                        missing)
        if unexpected:
            logger.warning("  Warm-start unexpected keys: %s", unexpected)
        warmstart_epoch = ckpt.get("epoch", "?")
        logger.info(
            "  Warm-start loaded successfully (trained epoch=%s)",
            warmstart_epoch,
        )

    # ------------------------------------------------------------------
    # 8b. Mimetic initialization (optional, via --mimetic-init)
    # ------------------------------------------------------------------
    if args.mimetic_init:
        logger.info("=" * 60)
        logger.info("MIMETIC INITIALIZATION — transferring teacher projections")
        logger.info("=" * 60)
        # Build a temporary base loader for calibration
        # (we use the train split to measure activation statistics)
        mimetic_init_from_teacher(
            student=student,
            teacher=teacher,
            train_loader=distill_loaders.get("train", list(distill_loaders.values())[0]),
            device=device,
            n_calibration_batches=10,
        )

    # ------------------------------------------------------------------
    # 9. Create multi-head distillation loss
    # ------------------------------------------------------------------
    distill_cfg = student_config.get("distillation", {})
    loss_cfg = student_config.get("loss", {})

    # CLI overrides for loss weight sweep experiments
    stim_w = args.stimulus_weight if args.stimulus_weight is not None else distill_cfg.get("stimulus_weight", 0.1)
    resp_w = args.response_weight if args.response_weight is not None else distill_cfg.get("response_weight", 0.1)
    # --distill-weight 0.0 disables KL divergence entirely (beta=0 ablation)
    distill_w = args.distill_weight if args.distill_weight is not None else distill_cfg.get("distill_weight", 0.5)

    # Hidden alignment weight: from CLI override or YAML config
    hidden_align_w = (
        args.hidden_align_weight
        if args.hidden_align_weight is not None
        else distill_cfg.get("hidden_align_weight", 0.0)
    )

    criterion = MultiHeadDistillationLoss(
        stimulus_weight=stim_w,
        response_weight=resp_w,
        hidden_align_weight=hidden_align_w,
        distill_weight=distill_w,
        distill_weight_min=distill_cfg.get("distill_weight_min", None),
        distill_schedule=distill_cfg.get("distill_schedule", None),
        reg_weight=distill_cfg.get("reg_weight", 0.001),
        reg_type=distill_cfg.get("reg_type", "l1"),
        log_input=loss_cfg.get("log_input", False),
    )

    # ------------------------------------------------------------------
    # 10. Create experiment folder
    # ------------------------------------------------------------------
    combined_config = {
        "teacher": teacher_config,
        "student": student_config,
        "data": data_config,
    }
    exp_dir = create_experiment(
        slug=args.slug,
        config=combined_config,
        command=" ".join(sys.argv),
        notes=f"Multi-head SNN distillation: "
              f"teacher={args.teacher_s3_slug or args.teacher_checkpoint}, "
              f"student_hidden={student_model_cfg.get('hidden_size', 256)}, "
              f"aux_heads={student_model_cfg.get('auxiliary_heads', [])}, "
              f"stim_wt={distill_cfg.get('stimulus_weight', 0.1)}, "
              f"resp_wt={distill_cfg.get('response_weight', 0.1)}",
    )
    logger.info("Experiment folder: %s", exp_dir)

    # ------------------------------------------------------------------
    # 11. Upload metadata to S3 (crash safety)
    # ------------------------------------------------------------------
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        try:
            from scripts.nrp_train import upload_experiment_metadata_to_s3
            upload_experiment_metadata_to_s3(exp_dir)
        except ImportError:
            logger.warning("Could not upload metadata to S3")

    # ------------------------------------------------------------------
    # 12. Initialize W&B
    # ------------------------------------------------------------------
    wandb_run = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            wandb_run = wandb.init(
                project="spike-prophecy",
                name=args.slug,
                config=combined_config,
                tags=["kosmos", "distillation", "multi-head", "snn"],
            )
            logger.info("W&B initialized: %s", wandb_run.url)
        except ImportError:
            logger.warning("wandb not installed — skipping")

    # ------------------------------------------------------------------
    # 13. Train with MultiHeadDistillTrainer
    # ------------------------------------------------------------------
    logger.info("Starting multi-head SNN distillation training...")

    # Wire S3 callbacks for crash-safe uploads
    checkpoint_callback = None
    metrics_callback = None
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        try:
            from scripts.nrp_train import (
                upload_checkpoint_to_s3,
                upload_metrics_to_s3,
            )
            checkpoint_callback = upload_checkpoint_to_s3
            # Trainer calls metrics_callback(epoch, history)
            metrics_callback = lambda epoch, history, exp=exp_dir: upload_metrics_to_s3(exp, epoch)
        except ImportError:
            pass

    trainer = MultiHeadDistillTrainer(
        model=student,
        train_loader=distill_loaders["train"],
        val_loader=distill_loaders["val"],
        config=student_config,
        device=device,
        criterion=criterion,
        exp_dir=exp_dir,
    )

    # Wire callbacks
    if checkpoint_callback:
        trainer.checkpoint_callback = checkpoint_callback
    if metrics_callback:
        trainer.metrics_callback = metrics_callback

    # -----------------------------------------------------------------
    # SGC config: wire warmdown and lambda_init to trainer for annealing.
    # Must be configured BEFORE trainer.train() so the trainer can read
    # _sgc_config during _train_one_epoch.
    # -----------------------------------------------------------------
    if student_model_cfg.get("sgc_enabled", False):
        trainer._sgc_config = {
            "warmdown_epochs": student_model_cfg.get("sgc_warmdown_epochs", 10),
            "lambda_init": student_model_cfg.get("sgc_lambda_init", 0.5),
        }
        logger.info(
            "SGC config: warmdown=%d, lambda_init=%.2f",
            trainer._sgc_config["warmdown_epochs"],
            trainer._sgc_config["lambda_init"],
        )

    # -----------------------------------------------------------------
    # Threshold annealing: ramp SNN thresholds from initial → calibrated
    # over training (only when mimetic init set calibrated values).
    # Must be configured BEFORE trainer.train() so the patched
    # _train_one_epoch is used from epoch 0.
    # -----------------------------------------------------------------
    if args.mimetic_init and hasattr(student, '_calibrated_threshold'):

        def _apply_threshold(model, threshold_val):
            """Apply threshold value to all spiking layers.

            Handles both snnTorch neurons (direct threshold attribute) and
            TI-LIF neurons (threshold_val buffer or threshold_param parameter).
            """
            import math
            for layer in model.spiking_layers:
                # TI-LIF: has threshold_val buffer or threshold_param
                if hasattr(layer, 'threshold_val'):
                    # Fixed threshold stored as buffer
                    layer.threshold_val.fill_(threshold_val)
                elif hasattr(layer, 'threshold_param'):
                    # Learnable threshold: stored as softplus input,
                    # so we need inverse softplus to set the desired value
                    inv_softplus = math.log(math.exp(threshold_val) - 1)
                    layer.threshold_param.data.fill_(inv_softplus)
                elif hasattr(layer, 'threshold'):
                    # snnTorch neurons: direct attribute
                    old_t = layer.threshold
                    if isinstance(old_t, (torch.Tensor, nn.Parameter)):
                        layer.threshold.data.fill_(threshold_val)
                    else:
                        layer.threshold = threshold_val

        # Patch _train_one_epoch to apply threshold annealing before each epoch
        original_train_one_epoch = trainer._train_one_epoch

        def _train_one_epoch_with_annealing():
            epoch = trainer._distill_epoch  # 0-indexed, before increment
            total_ep = trainer._total_epochs
            progress = min(epoch / max(total_ep - 1, 1), 1.0)
            current_threshold = (
                student._initial_threshold
                + (student._calibrated_threshold - student._initial_threshold)
                * progress
            )
            _apply_threshold(student, current_threshold)
            logger.info(
                "  Threshold annealing: %.4f (epoch %d/%d, range [%.4f, %.4f])",
                current_threshold, epoch + 1, total_ep,
                student._initial_threshold, student._calibrated_threshold,
            )
            return original_train_one_epoch()

        trainer._train_one_epoch = _train_one_epoch_with_annealing
        logger.info(
            "Threshold annealing enabled: %.4f → %.4f over %d epochs",
            student._initial_threshold, student._calibrated_threshold,
            student_config.get("training", {}).get("epochs", 50),
        )

    history = trainer.train()

    # ------------------------------------------------------------------
    # 14. Log final metrics
    # ------------------------------------------------------------------
    if wandb_run is not None:
        try:
            if history and len(history) > 0:
                last = history[-1] if isinstance(history, list) else history
                for key, value in last.items():
                    if isinstance(value, (int, float)):
                        wandb_run.summary[key] = value
            wandb_run.finish()
        except Exception as e:
            logger.warning("W&B logging failed: %s", e)

    # ------------------------------------------------------------------
    # 15. Upload full experiment to S3
    # ------------------------------------------------------------------
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        logger.info("Uploading experiment to S3...")
        try:
            from scripts.nrp_train import upload_experiment_to_s3
            upload_experiment_to_s3(exp_dir)
        except ImportError:
            logger.warning("Could not import S3 upload — skipping")

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(
        "MULTI-HEAD SNN DISTILLATION COMPLETE — %.1f minutes", elapsed / 60,
    )
    logger.info(
        "Student: %d params | Teacher: %d params | Ratio: %.2fx",
        student_params, teacher_params, student_params / max(teacher_params, 1),
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
