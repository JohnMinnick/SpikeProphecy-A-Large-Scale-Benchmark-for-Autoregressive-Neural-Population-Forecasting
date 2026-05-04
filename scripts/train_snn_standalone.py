"""
Train a standalone Student SNN directly on ground truth (no teacher).

This script is a control experiment for the distillation approach:
it trains the same StudentSNN architecture with pure Poisson NLL on
raw spike counts, without any teacher guidance or KL divergence.

If the standalone SNN underperforms the distilled SNN, it confirms
that the teacher's soft targets provide value beyond the raw data.

Usage:
    # Local:
    python scripts/train_snn_standalone.py \
        --data-config configs/data/steinmetz_multi_nrp_50ms_no_cov.yaml \
        --student-config configs/student/standalone_snn.yaml \
        --slug snn-standalone-smoke --epochs 3

    # NRP:
    python scripts/train_snn_standalone.py \
        --data-config configs/data/steinmetz_multi_nrp_50ms_no_cov.yaml \
        --student-config configs/student/standalone_snn.yaml \
        --slug snn-standalone-v1
"""

import argparse
import gc
import json as _json
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.multi_session_loader import (
    preprocess_and_cache,
    create_dataloaders,
    MaskedSpikeCountDataset,
    pad_to_channels,
    build_channel_mask,
)
from src.models.student import StudentSNN
from src.train.trainer import Trainer
from src.eval.metrics import poisson_nll, pearson_r, mae, mse
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
logger = logging.getLogger("train_snn_standalone")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone SNN training."""
    parser = argparse.ArgumentParser(
        description="Train Student SNN directly on spike data (no teacher)."
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
        "--slug", type=str, default="snn-standalone",
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
        "--dry-run", action="store_true",
        help="Validate config only, don't train.",
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Path to checkpoint to warm-restart from (model weights only, "
             "fresh optimizer). Supports local path or S3 path like "
             "s3://bucket/path/to/best_model.pt",
    )
    # Multi-head behavioral decoding options
    parser.add_argument(
        "--auxiliary-heads", nargs="*", default=None,
        help="Enable multi-head behavioral decode. List head names: "
             "'stimulus' (16-class contrast), 'response' (3-class choice). "
             "Example: --auxiliary-heads stimulus response",
    )
    parser.add_argument(
        "--stimulus-weight", type=float, default=0.1,
        help="Loss weight for stimulus classification head (default 0.1).",
    )
    parser.add_argument(
        "--response-weight", type=float, default=0.1,
        help="Loss weight for response classification head (default 0.1).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override seed in student_config['training']['seed']. "
             "Used for multi-seed sweeps (NeurIPS 2026 reviewer ask).",
    )
    return parser.parse_args()


# =========================================================================
# Behavioral data loading and dataset wrapping (multi-head mode)
# =========================================================================

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
            base_dataset: MaskedSpikeCountDataset returning (x, y, mask).
            behavior_arrays: Dict with keys left_contrast, right_contrast,
                response_choice, trial_active — all shape (T_full,) for the
                full session recording. None if no behavior available.
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
    mask_batch = torch.stack(masks, dim=0)

    beh_batch = {
        key: torch.stack([b[key] for b in behs], dim=0)
        for key in behs[0].keys()
    }

    return x_batch, y_batch, beh_batch, mask_batch


def load_session_behaviors_unified(
    data_config: dict,
    multi_meta: dict,
    holdout_frac: float = 0.2,
    holdout_seed: int = 42,
) -> dict:
    """
    Pre-extract behavioral data from all sessions (Steinmetz NWB or IBL).

    Routes to the correct extraction function based on source type.
    For combined datasets, loads Steinmetz NWB behavior for the first N
    sessions and IBL behavior for the remaining sessions.

    Args:
        data_config: Data config dict with source type and paths.
        multi_meta: Metadata from preprocess_and_cache().
        holdout_frac: Fraction of trials to hold out per session.
        holdout_seed: Random seed for trial holdout.

    Returns:
        Dict mapping session_id (str) -> behavior dict.
    """
    source_type = data_config.get("source", {}).get("type", "nwb_multi")
    bin_width_ms = data_config.get("bin_width_ms", 50.0)
    bin_width_s = bin_width_ms / 1000.0
    sessions_meta = multi_meta.get("sessions", [])

    session_behaviors = {}
    total_train_trials = 0
    total_eval_trials = 0

    # Determine which sessions are NWB (Steinmetz) vs IBL
    # For combined datasets, sessions are ordered: Steinmetz first, then IBL.
    # The session metadata has a 'file' field: NWB paths for Steinmetz,
    # IBL EIDs (UUIDs) for IBL sessions.
    for si, sess_info in enumerate(sessions_meta):
        session_id = f"session_{si:03d}"
        n_bins = sess_info.get("num_bins", 0)
        if n_bins == 0:
            continue

        # Compute bin edges for this session
        bin_edges = np.arange(n_bins + 1) * bin_width_s

        try:
            # Detect source type for this session:
            # NWB files end in .nwb; IBL EIDs are UUIDs (contain dashes)
            file_id = sess_info.get("file", "")
            is_ibl_session = (
                not str(file_id).endswith(".nwb")
                and "-" in str(file_id)
                and len(str(file_id)) > 30
            )

            if is_ibl_session:
                # IBL session: use ONE API-based behavior extraction
                from src.data.ibl_behavior_loader import (
                    extract_ibl_trial_stimuli,
                )
                ibl_cache_dir = data_config.get(
                    "ibl", {},
                ).get("cache_dir", "data/raw/ibl")
                behavior = extract_ibl_trial_stimuli(
                    eid=str(file_id),
                    bin_edges=bin_edges,
                    cache_dir=ibl_cache_dir,
                )
            else:
                # Steinmetz NWB session: use h5py-based behavior extraction
                from src.data.behavior_loader import extract_trial_stimuli

                # Find the NWB file path
                source_glob = data_config.get(
                    "source", {},
                ).get("glob", "data/raw/*.nwb")
                nwb_glob_path = Path(source_glob)
                nwb_files = sorted(nwb_glob_path.parent.glob(nwb_glob_path.name))

                if si < len(nwb_files):
                    behavior = extract_trial_stimuli(
                        str(nwb_files[si]), bin_edges,
                    )
                else:
                    logger.warning(
                        "Session %s: no NWB file at index %d, skipping behavior",
                        session_id, si,
                    )
                    continue

            # ---------------------------------------------------------------
            # Trial-level holdout: randomly hold out ~holdout_frac of trials
            # ---------------------------------------------------------------
            trial_indices = behavior["trial_index"]
            unique_trials = np.unique(trial_indices[trial_indices >= 0])
            n_trials = len(unique_trials)

            if n_trials == 0:
                logger.warning(
                    "Session %s: no valid trials found, skipping",
                    session_id,
                )
                continue

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

            # Convert to float32 arrays
            session_behaviors[session_id] = {
                k: v.astype(np.float32) if hasattr(v, 'dtype') and v.dtype != np.float32 else v
                for k, v in behavior.items()
            }

            n_active = int(behavior["trial_active"].sum())
            logger.info(
                "Session %s: %d bins, %d active, %d/%d trials (train/eval), %s",
                session_id, n_bins, n_active,
                len(train_trials), len(eval_trials),
                "IBL" if is_ibl_session else "NWB",
            )

        except Exception as e:
            logger.warning(
                "Failed to load behavior for %s: %s", session_id, e,
            )

    logger.info(
        "Loaded behavioral data for %d/%d sessions "
        "(%d train trials, %d eval trials)",
        len(session_behaviors), len(sessions_meta),
        total_train_trials, total_eval_trials,
    )
    return session_behaviors


def _encode_stimulus_class(
    left_contrast: torch.Tensor,
    right_contrast: torch.Tensor,
) -> torch.Tensor:
    """
    Encode (left, right) contrast pair into a 16-class index.

    Each contrast ∈ {0, 0.25, 0.5, 1.0}, so 4 × 4 = 16 classes.
    Class index = left_bin * 4 + right_bin, where:
        bin 0 = 0.0, bin 1 = 0.25, bin 2 = 0.5, bin 3 = 1.0

    Args:
        left_contrast: (batch,) left contrast values.
        right_contrast: (batch,) right contrast values.

    Returns:
        class_idx: (batch,) int64 class indices in [0, 15].
    """
    # Discretize contrasts to bin indices
    contrast_bins = torch.tensor([0.0, 0.25, 0.5, 1.0], device=left_contrast.device)

    def _to_bin(vals):
        # Find closest bin for each value
        diffs = (vals.unsqueeze(-1) - contrast_bins.unsqueeze(0)).abs()
        return diffs.argmin(dim=-1)

    left_bin = _to_bin(left_contrast)
    right_bin = _to_bin(right_contrast)
    return (left_bin * 4 + right_bin).long()


def main() -> None:
    """Main standalone SNN training pipeline."""
    start_time = time.time()
    args = parse_args()

    # Determine if multi-head mode is active
    multi_head = args.auxiliary_heads is not None and len(args.auxiliary_heads) > 0

    logger.info("=" * 60)
    if multi_head:
        logger.info("MULTI-HEAD STANDALONE SNN TRAINING — SpikeProphecy")
    else:
        logger.info("STANDALONE SNN TRAINING — SpikeProphecy (Control)")
    logger.info("=" * 60)
    logger.info("  Student config:     %s", args.student_config)
    logger.info("  Data config:        %s", args.data_config)
    logger.info("  Slug:               %s", args.slug)
    if multi_head:
        logger.info("  Auxiliary heads:    %s", args.auxiliary_heads)
        logger.info("  Stimulus weight:    %.3f", args.stimulus_weight)
        logger.info("  Response weight:    %.3f", args.response_weight)
    logger.info("  NOTE: No teacher — direct Poisson NLL on ground truth")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load configs
    # ------------------------------------------------------------------
    data_config = load_config(args.data_config)
    student_config = load_config(args.student_config)

    # Apply CLI overrides
    if args.epochs is not None:
        student_config.setdefault("training", {})["epochs"] = args.epochs
    if args.lr is not None:
        student_config.setdefault("training", {})["learning_rate"] = args.lr

    if args.dry_run:
        logger.info("[DRY RUN] Configs validated. Exiting.")
        return

    # ------------------------------------------------------------------
    # 2. Download data from S3 if NRP mode
    # ------------------------------------------------------------------
    # Check source type to route to the correct S3 download path.
    # IBL data uses pre-cached .npy arrays from a different S3 prefix
    # than the Steinmetz .nwb files. This must be checked BEFORE
    # calling any download function.
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
                # This mirrors the logic in nrp_train.py main() lines 488-542.
                #
                # Auto-detect S3 prefix from config's ibl.tag field:
                #   "repeated_site" → .../ibl-repeated-site/
                #   "combined"      → .../combined-steinmetz-ibl/
                # Env vars override if explicitly set.
                ibl_tag = data_config.get("ibl", {}).get("tag", "repeated_site")
                tag_to_prefix = {
                    "repeated_site": "jrm/spike-prophecy/inputs/ibl-repeated-site",
                    "combined": "jrm/spike-prophecy/inputs/combined-steinmetz-ibl",
                }
                tag_to_cache = {
                    "repeated_site": "ibl_repeated_site_cache",
                    "combined": "combined_steinmetz_ibl_cache",
                }
                default_prefix = tag_to_prefix.get(
                    ibl_tag, f"jrm/spike-prophecy/inputs/{ibl_tag}"
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

    # ------------------------------------------------------------------
    # 3. Set seed for reproducibility (CLI --seed overrides config)
    # ------------------------------------------------------------------
    if getattr(args, "seed", None) is not None:
        student_config.setdefault("training", {})["seed"] = args.seed
        logger.info("Seed override applied: seed=%d", args.seed)
    seed = student_config.get("training", {}).get("seed", 42)
    seed_everything(seed)

    # ------------------------------------------------------------------
    # 4. Preprocess data and create DataLoaders
    # ------------------------------------------------------------------
    # Dispatch on source.type: IBL data uses pre-cached .npy arrays
    # (downloaded from S3 in step 2), while NWB data needs NWB→cache
    # conversion via preprocess_and_cache(). This mirrors train_teacher.py.
    logger.info("Preprocessing multi-session data...")

    if source_type == "ibl":
        # IBL source: use pre-cached .npy files from S3 if metadata exists,
        # otherwise fall back to preprocess_and_cache_ibl() (ONE API download).
        import json as _json

        # Determine cache dir name from ibl.tag (matches S3 download step 2)
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
            # S3-downloaded cache: skip re-processing entirely
            with open(_meta_path, "r", encoding="utf-8") as _mf:
                multi_meta = _json.load(_mf)
            cache_dir = ibl_cache_dir
            logger.info(
                "Found existing IBL cache metadata: %d sessions, M_max=%d "
                "— skipping re-processing",
                multi_meta["num_sessions"], multi_meta["m_max"],
            )
        else:
            # No S3 cache: run IBL preprocessing from ONE API
            from src.data.ibl_data_loader import preprocess_and_cache_ibl
            cache_dir, multi_meta = preprocess_and_cache_ibl(
                data_config, cache_dir=str(ibl_cache_dir),
            )
    else:
        # NWB source (Steinmetz): standard NWB → cache preprocessing
        cache_dir, multi_meta = preprocess_and_cache(data_config)

    m_max = multi_meta["m_max"]
    logger.info("Data ready: M_max=%d, %d sessions", m_max, multi_meta["num_sessions"])

    # Only build full dataloaders for dynamics-only mode.
    # Multi-head mode loads sessions lazily to avoid OOM on 105-session
    # combined datasets (~1998 channels × 15K bins × 105 sessions).
    if not multi_head:
        base_loaders = create_dataloaders(cache_dir, multi_meta, data_config)
    else:
        base_loaders = None
        logger.info(
            "Multi-head mode: skipping create_dataloaders() — "
            "will load sessions lazily to conserve RAM"
        )

    # ------------------------------------------------------------------
    # 5. Resolve device
    # ------------------------------------------------------------------
    device = resolve_device()
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    # 6. Create Student SNN (same architecture as distilled version)
    # ------------------------------------------------------------------
    logger.info("Creating StudentSNN (random init, NO teacher)...")
    student_model_cfg = student_config.get("model", {})

    # Merge auxiliary_heads from CLI into model config if specified
    effective_aux_heads = args.auxiliary_heads if multi_head else (
        student_model_cfg.get("auxiliary_heads", None)
    )

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
        auxiliary_heads=effective_aux_heads,
    )
    student.to(device)

    student_params = sum(p.numel() for p in student.parameters())
    logger.info("Student created: %d params (standalone, no teacher)", student_params)

    # ------------------------------------------------------------------
    # 6b. Warm restart: load model weights from checkpoint (fresh optimizer)
    # ------------------------------------------------------------------
    if args.resume_from:
        logger.info("=" * 60)
        logger.info("WARM RESTART from checkpoint: %s", args.resume_from)
        logger.info("Loading model weights only — fresh optimizer + scheduler")
        logger.info("=" * 60)

        ckpt_path = args.resume_from

        # Download from S3 if path starts with s3://
        if ckpt_path.startswith("s3://"):
            import boto3
            from botocore.config import Config as BotoConfig
            local_ckpt = Path("/tmp/warm_restart_ckpt.pt")

            # Parse S3 path: s3://bucket/key
            s3_parts = ckpt_path.replace("s3://", "").split("/", 1)
            bucket_name = s3_parts[0]
            s3_key = s3_parts[1]

            # Use boto3.resource with same config as nrp/s3_utils.py
            # (trailing slash on endpoint URL is required for Ceph)
            s3_endpoint = os.environ.get(
                "S3_ENDPOINT", "http://rook-ceph-rgw-nautiluss3.rook"
            )
            if not s3_endpoint.endswith("/"):
                s3_endpoint += "/"

            s3_res = boto3.resource(
                "s3",
                endpoint_url=s3_endpoint,
                aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                config=BotoConfig(
                    retries={"max_attempts": 5, "mode": "adaptive"},
                    connect_timeout=30,
                    read_timeout=60,
                ),
            )
            logger.info("Downloading checkpoint from S3: %s/%s", bucket_name, s3_key)
            s3_res.Bucket(bucket_name).download_file(s3_key, str(local_ckpt))
            ckpt_path = str(local_ckpt)
            logger.info("Downloaded checkpoint to %s", ckpt_path)

        # Load checkpoint and extract model weights only
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        student.load_state_dict(checkpoint["model_state_dict"])
        resume_epoch = checkpoint.get("epoch", "?")
        resume_val_loss = checkpoint.get("best_val_loss", "?")
        logger.info(
            "Loaded model weights from epoch %s (best_val_loss=%s)",
            resume_epoch, resume_val_loss,
        )
        logger.info("Optimizer and scheduler will be initialized fresh.")

    # ------------------------------------------------------------------
    # 7. Create experiment folder
    # ------------------------------------------------------------------
    combined_config = {
        "student": student_config,
        "data": data_config,
        "mode": "standalone_snn_no_teacher",
    }
    exp_dir = create_experiment(
        slug=args.slug,
        config=combined_config,
        command=" ".join(sys.argv),
        notes=f"Standalone SNN (no teacher distillation), "
              f"hidden={student_model_cfg.get('hidden_size', 256)}, "
              f"pure Poisson NLL on ground truth",
    )
    logger.info("Experiment folder: %s", exp_dir)

    # ------------------------------------------------------------------
    # 8. Upload metadata to S3 (crash safety)
    # ------------------------------------------------------------------
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        try:
            sys.path.insert(0, str(PROJECT_ROOT / "nrp"))
            from s3_utils import upload_files
            # Upload config.yaml as early metadata
            config_path = exp_dir / "config.yaml"
            if config_path.exists():
                upload_files(
                    "jrm/spike-prophecy/outputs", args.slug,
                    str(config_path),
                )
                logger.info("Uploaded metadata to S3")
        except Exception as e:
            logger.warning("Could not upload metadata to S3: %s", e)

    # ------------------------------------------------------------------
    # 9. Initialize W&B if available
    # ------------------------------------------------------------------
    wandb_run = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            wandb_run = wandb.init(
                project="spike-prophecy",
                name=args.slug,
                config=combined_config,
                tags=["standalone-snn", "control", "no-teacher"],
            )
            logger.info("W&B initialized: %s", wandb_run.url)
        except ImportError:
            logger.warning("wandb not installed — skipping")

    # ------------------------------------------------------------------
    # 10. Train (standard or multi-head)
    # ------------------------------------------------------------------
    if not multi_head:
        # =============================================================
        # DYNAMICS-ONLY MODE: use standard Trainer (original behavior)
        # =============================================================
        logger.info("Starting standalone SNN training (Poisson NLL, no teacher)...")

        trainer = Trainer(
            model=student,
            train_loader=base_loaders["train"],
            val_loader=base_loaders["val"],
            config=student_config,
            device=device,
            exp_dir=exp_dir,
        )
        history = trainer.train()
    else:
        # =============================================================
        # MULTI-HEAD MODE: dynamics + behavioral decode
        # =============================================================
        logger.info("Starting MULTI-HEAD standalone SNN training...")
        logger.info("  Heads: %s", effective_aux_heads)

        # 10a. Load behavioral data for all sessions
        session_behaviors = load_session_behaviors_unified(
            data_config, multi_meta,
            holdout_frac=0.2,
            holdout_seed=42,
        )

        # 10b. Training hyperparameters
        train_cfg = student_config.get("training", {})
        epochs = train_cfg.get("epochs", 50)
        lr = train_cfg.get("learning_rate", 0.001)
        weight_decay = train_cfg.get("weight_decay", 1e-4)
        grad_clip = train_cfg.get("grad_clip_norm", 1.0)
        spike_reg = train_cfg.get("spike_reg_lambda", 0.0001)
        stim_weight = args.stimulus_weight
        resp_weight = args.response_weight

        optimizer = torch.optim.AdamW(
            student.parameters(), lr=lr, weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1),
        )

        # Loss functions for auxiliary heads
        ce_loss_fn = nn.CrossEntropyLoss(reduction="none")

        # Training + validation loop
        best_val_r = -float("inf")
        history = []

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            student.train()

            # ---- Train epoch (session-by-session with behavior) ----
            total_dyn_loss = 0.0
            total_stim_loss = 0.0
            total_resp_loss = 0.0
            total_reg_loss = 0.0
            n_train_batches = 0

            import random
            session_indices = list(range(multi_meta["num_sessions"]))
            random.shuffle(session_indices)

            for sess_idx in session_indices:
                sess_info = multi_meta["sessions"][sess_idx]
                session_id = f"session_{sess_idx:03d}"

                # Load cached count matrix
                npy_path = Path(cache_dir) / f"session_{sess_idx:03d}.npy"
                if not npy_path.exists():
                    continue
                counts_u8 = np.load(npy_path)

                # Get train split slice
                split_bounds = sess_info.get("split_boundaries", {})
                train_end = split_bounds.get(
                    "train_end",
                    int(counts_u8.shape[1] * 0.7),
                )
                split_start = 0
                split_end = train_end
                split_len = split_end - split_start
                history_bins = multi_meta.get("history_bins", 10)
                if split_len <= history_bins:
                    del counts_u8
                    continue

                counts_split = counts_u8[:, split_start:split_end].astype(np.int32)
                del counts_u8

                # Pad to M_max
                m_i = counts_split.shape[0]
                padded = pad_to_channels(counts_split, m_max)
                del counts_split

                # Build mask
                mask = build_channel_mask(m_i, m_max)
                session_masks = mask.reshape(1, -1)
                mask_index = np.zeros(padded.shape[1], dtype=np.int32)

                # Create base spike dataset
                base_ds = MaskedSpikeCountDataset(
                    spike_counts=padded,
                    mask_index=mask_index,
                    session_masks=session_masks,
                    history_bins=history_bins,
                    output_channels=m_max,
                )
                del padded, mask_index

                if len(base_ds) == 0:
                    del base_ds
                    continue

                # Get behavior for this session
                beh_arrays = session_behaviors.get(session_id, None)

                # Wrap with behavior
                aug_ds = BehaviorAugmentedDataset(
                    base_dataset=base_ds,
                    behavior_arrays=beh_arrays,
                    split_start=split_start,
                    history_bins=history_bins,
                )

                # DataLoader
                n_workers = 0 if platform.system() == "Windows" else 2
                dl = DataLoader(
                    aug_ds,
                    batch_size=train_cfg.get("batch_size", 512),
                    shuffle=True,
                    num_workers=n_workers,
                    pin_memory=(platform.system() != "Windows"),
                    drop_last=True,
                    collate_fn=_collate_with_behavior,
                )

                # Process batches
                for x, y, behavior, ch_mask in dl:
                    x = x.to(device)
                    y = y.to(device)
                    ch_mask = ch_mask.to(device)
                    beh_dev = {
                        k: v.to(device) for k, v in behavior.items()
                    }

                    # Forward pass — multi-head returns dict
                    output = student(x)
                    if isinstance(output, tuple):
                        output = {"rates": output[0], "spikes": output[1]}

                    rates = output["rates"]
                    spikes = output["spikes"]

                    # --- Dynamics loss (masked Poisson NLL) ---
                    eps = 1e-8
                    per_elem = rates - y * torch.log(rates + eps)
                    dyn_loss = (per_elem * ch_mask).sum() / ch_mask.sum().clamp(min=1.0)

                    # --- Spike regularization ---
                    reg_loss = spike_reg * spikes.abs().mean()

                    # --- Auxiliary losses (trial-masked) ---
                    s_loss = torch.tensor(0.0, device=device)
                    r_loss = torch.tensor(0.0, device=device)

                    # Only compute aux losses on trial-active bins
                    # with behavior_train_mask (excludes held-out trials)
                    train_mask = beh_dev.get(
                        "behavior_train_mask",
                        beh_dev["trial_active"],
                    )
                    active = train_mask > 0.5

                    if active.sum() > 0 and "stimulus" in output:
                        stim_logits = output["stimulus"][active]  # (N_active, 16)
                        stim_targets = _encode_stimulus_class(
                            beh_dev["left_contrast"][active],
                            beh_dev["right_contrast"][active],
                        )
                        s_loss = stim_weight * ce_loss_fn(
                            stim_logits, stim_targets,
                        ).mean()

                    if active.sum() > 0 and "response" in output:
                        resp_logits = output["response"][active]  # (N_active, 3)
                        # Map {-1, 0, +1} → {0, 1, 2}
                        resp_targets = (
                            beh_dev["response_choice"][active] + 1
                        ).long()
                        r_loss = resp_weight * ce_loss_fn(
                            resp_logits, resp_targets,
                        ).mean()

                    # Total loss
                    loss = dyn_loss + reg_loss + s_loss + r_loss

                    optimizer.zero_grad()
                    loss.backward()
                    if grad_clip > 0:
                        nn.utils.clip_grad_norm_(
                            student.parameters(), grad_clip,
                        )
                    optimizer.step()

                    total_dyn_loss += dyn_loss.item()
                    total_stim_loss += s_loss.item()
                    total_resp_loss += r_loss.item()
                    total_reg_loss += reg_loss.item()
                    n_train_batches += 1

                # Free memory
                del dl, aug_ds, base_ds
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            scheduler.step()

            # ---- Validation (lazy per-session eval) ----
            # Load sessions one-at-a-time to avoid OOM.
            # Accumulate per-session Pearson r then neuron-weight avg.
            student.eval()
            history_bins = multi_meta.get("history_bins", 10)
            session_r_vals = []  # (r, n_neurons) per session
            session_losses = []  # (loss, n_samples) per session

            with torch.no_grad():
                for sess_idx in range(multi_meta["num_sessions"]):
                    sess_info = multi_meta["sessions"][sess_idx]
                    npy_path = Path(cache_dir) / f"session_{sess_idx:03d}.npy"
                    if not npy_path.exists():
                        continue

                    counts_u8 = np.load(npy_path)

                    # Val split: from val_start to end
                    split_bounds = sess_info.get("split_boundaries", {})
                    val_start = split_bounds.get(
                        "val_start",
                        int(counts_u8.shape[1] * 0.7),
                    )
                    val_end = counts_u8.shape[1]
                    if val_end - val_start <= history_bins:
                        del counts_u8
                        continue

                    counts_val = counts_u8[:, val_start:val_end].astype(np.int32)
                    m_i = counts_val.shape[0]
                    del counts_u8

                    padded_val = pad_to_channels(counts_val, m_max)
                    del counts_val

                    mask = build_channel_mask(m_i, m_max)
                    masks = mask.reshape(1, -1)
                    mask_idx = np.zeros(padded_val.shape[1], dtype=np.int32)

                    val_ds = MaskedSpikeCountDataset(
                        spike_counts=padded_val,
                        mask_index=mask_idx,
                        session_masks=masks,
                        history_bins=history_bins,
                        output_channels=m_max,
                    )
                    del padded_val, mask_idx

                    if len(val_ds) == 0:
                        del val_ds
                        continue

                    n_workers = 0 if platform.system() == "Windows" else 2
                    val_dl = DataLoader(
                        val_ds,
                        batch_size=train_cfg.get("batch_size", 512),
                        shuffle=False,
                        num_workers=n_workers,
                        pin_memory=False,
                    )

                    sess_preds = []
                    sess_targets = []
                    for batch in val_dl:
                        x_v, y_v = batch[0].to(device), batch[1].to(device)
                        out = student(x_v)
                        if isinstance(out, dict):
                            r_v = out["rates"]
                        elif isinstance(out, tuple):
                            r_v = out[0]
                        else:
                            r_v = out
                        sess_preds.append(r_v.cpu())
                        sess_targets.append(y_v.cpu())

                    # Per-session metrics
                    sp = torch.cat(sess_preds, dim=0)
                    st = torch.cat(sess_targets, dim=0)
                    s_r = float(pearson_r(sp, st))
                    s_loss = float(poisson_nll(sp, st))
                    session_r_vals.append((s_r, m_i))
                    session_losses.append((s_loss, len(val_ds)))

                    # Free
                    del val_dl, val_ds, sess_preds, sess_targets, sp, st
                    gc.collect()

            # Neuron-weighted average
            total_neurons = sum(n for _, n in session_r_vals)
            if total_neurons > 0:
                val_r = sum(r * n for r, n in session_r_vals) / total_neurons
            else:
                val_r = 0.0
            total_samples = sum(n for _, n in session_losses)
            if total_samples > 0:
                val_loss = sum(l * n for l, n in session_losses) / total_samples
            else:
                val_loss = 0.0
            val_mae_v = 0.0  # Skip MAE for speed; val_r is primary

            avg_dyn = total_dyn_loss / max(n_train_batches, 1)
            avg_stim = total_stim_loss / max(n_train_batches, 1)
            avg_resp = total_resp_loss / max(n_train_batches, 1)

            epoch_time = time.time() - epoch_start
            logger.info(
                "Epoch %d/%d (%.1fs) | train: dyn=%.4f stim=%.4f resp=%.4f | "
                "val_r=%.4f val_loss=%.4f val_mae=%.4f",
                epoch, epochs, epoch_time,
                avg_dyn, avg_stim, avg_resp,
                val_r, val_loss, val_mae_v,
            )

            # Checkpoint best model
            if val_r > best_val_r:
                best_val_r = val_r
                if exp_dir:
                    ckpt_path = exp_dir / "best_model.pt"
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": student.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_r": best_val_r,
                        "best_val_loss": val_loss,
                    }, ckpt_path)
                    logger.info(
                        "  ✓ New best val_r=%.4f — saved to %s",
                        best_val_r, ckpt_path,
                    )

            history.append({
                "epoch": epoch,
                "train_dyn_loss": avg_dyn,
                "train_stim_loss": avg_stim,
                "train_resp_loss": avg_resp,
                "val_loss": val_loss,
                "val_r": val_r,
                "val_mae": val_mae_v,
            })

            # Save metrics JSON incrementally
            if exp_dir:
                metrics_path = exp_dir / "metrics.json"
                import json
                with open(metrics_path, "w") as f:
                    json.dump(history, f, indent=2)

        logger.info(
            "Multi-head training complete: best val_r=%.4f", best_val_r,
        )

    # ------------------------------------------------------------------
    # 10b. Population metrics (co-BPS, calibration, PSTH R²)
    # ------------------------------------------------------------------
    pop_metrics = {}
    pop_scalars = {}
    if not multi_head:
        try:
            logger.info("Computing population metrics...")
            pop_metrics = trainer.evaluate_population(base_loaders["val"])
            pop_scalars = {
                k: v for k, v in pop_metrics.items()
                if isinstance(v, (int, float, type(None)))
            }
            logger.info("Population metrics:")
            for k, v in sorted(pop_scalars.items()):
                if isinstance(v, float):
                    logger.info("  %s = %.4f", k, v)
        except Exception as e:
            logger.warning("Population metrics failed: %s", e)

    # ------------------------------------------------------------------
    # 11. Log final metrics
    # ------------------------------------------------------------------
    if wandb_run is not None:
        try:
            if history and len(history) > 0:
                last = history[-1] if isinstance(history, list) else history
                for key, value in last.items():
                    if isinstance(value, (int, float)):
                        wandb_run.summary[key] = value
            # Log population metrics to W&B summary
            for key, value in pop_scalars.items():
                if isinstance(value, (int, float)):
                    wandb_run.summary[f"pop_{key}"] = value
            wandb_run.finish()
        except Exception as e:
            logger.warning("W&B logging failed: %s", e)

    # ------------------------------------------------------------------
    # 12. Upload full experiment to S3
    # ------------------------------------------------------------------
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        logger.info("Uploading experiment to S3...")
        try:
            sys.path.insert(0, str(PROJECT_ROOT / "nrp"))
            from s3_utils import upload_files
            # Upload all files in experiment directory
            for f in exp_dir.rglob("*"):
                if f.is_file():
                    upload_files(
                        "jrm/spike-prophecy/outputs", args.slug,
                        str(f),
                    )
            logger.info("Experiment uploaded to S3")
        except Exception as e:
            logger.warning("Could not upload to S3: %s", e)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("STANDALONE SNN TRAINING COMPLETE — %.1f minutes", elapsed / 60)
    logger.info("  Student: %d params (no teacher)", student_params)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
