"""
Generate Mamba prediction arrays for all sessions and upload to S3.

Runs Mamba inference on each session's validation split and saves
per-session .npz files containing ground truth and predicted rates.
These arrays are used by the local IFER figure generation script.

This script is designed to run on NRP (Linux + CUDA) since Mamba
requires the mamba-ssm package which only compiles on Linux.

Usage (NRP):
    python scripts/generate_prediction_arrays.py \
        --teacher-s3-slug 2026-03-26_baseline-mamba-v12 \
        --teacher-config configs/teacher/nrp_teacher_mamba.yaml \
        --data-config configs/data/steinmetz_multi_nrp_50ms_no_cov.yaml \
        --upload-slug ifer-prediction-arrays
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.multi_session_loader import preprocess_and_cache
from src.models.teacher import create_teacher_model
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.seed import seed_everything

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_prediction_arrays")

# ---------------------------------------------------------------------------
# S3 Configuration
# ---------------------------------------------------------------------------
S3_BUCKET = "braingeneersdev"
S3_OUTPUT_PREFIX = "jrm/spike-prophecy/outputs"
S3_CHECKPOINT_PREFIX = "jrm/spike-prophecy/outputs"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for prediction array generation."""
    parser = argparse.ArgumentParser(
        description="Generate Mamba prediction arrays for all sessions.",
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
        "--teacher-s3-slug", type=str, required=True,
        help="S3 experiment slug for the Mamba checkpoint "
             "(e.g., '2026-03-26_baseline-mamba-v12').",
    )
    parser.add_argument(
        "--upload-slug", type=str, default="ifer-prediction-arrays",
        help="S3 slug for uploading prediction arrays. "
             "Default: 'ifer-prediction-arrays'.",
    )
    parser.add_argument(
        "--window-length", type=int, default=200,
        help="Number of time bins to use from the validation split. "
             "Default: 200 (10s at 50ms bins).",
    )
    parser.add_argument(
        "--sessions", type=int, nargs="*", default=None,
        help="Specific sessions to process (default: all).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility. Default: 42.",
    )
    return parser.parse_args()


def download_checkpoint_from_s3(slug: str, local_dir: str = "/tmp") -> Path:
    """
    Download a Mamba checkpoint from S3.

    Args:
        slug: Experiment slug (e.g., '2026-03-26_baseline-mamba-v12').
        local_dir: Local directory to save the checkpoint.

    Returns:
        Path to the downloaded checkpoint file.
    """
    from nrp.s3_utils import bucket, retry_with_backoff

    local_path = Path(local_dir) / f"{slug}_best_model.pt"
    if local_path.exists():
        logger.info("Checkpoint already exists: %s", local_path)
        return local_path

    s3_key = f"{S3_CHECKPOINT_PREFIX}/{slug}/best_model.pt"
    logger.info("Downloading checkpoint: s3://%s/%s", S3_BUCKET, s3_key)
    retry_with_backoff(lambda: bucket.download_file(s3_key, str(local_path)))
    size_mb = local_path.stat().st_size / 1e6
    logger.info("Downloaded: %s (%.1f MB)", local_path, size_mb)
    return local_path


@torch.no_grad()
def predict_session(
    model: torch.nn.Module,
    counts: np.ndarray,
    m_max: int,
    history_bins: int,
    start_bin: int,
    window_length: int,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Run Mamba inference over a validation window for one session.

    Args:
        model: Frozen Mamba teacher model.
        counts: Raw spike counts (M_i, T_total) — unpadded.
        m_max: Global max neuron count (for zero-padding input).
        history_bins: Number of history bins (T).
        start_bin: First target bin (must be >= history_bins).
        window_length: Number of bins to predict.
        device: Compute device (CUDA).
        batch_size: Mini-batch size for inference.

    Returns:
        predicted_rates: (window_length, M_i) array of predicted rates.
    """
    model.eval()
    m_i = counts.shape[0]

    # Zero-pad to M_max for the shared-head model
    if m_i < m_max:
        padded = np.zeros((m_max, counts.shape[1]), dtype=np.float32)
        padded[:m_i, :] = counts
    else:
        padded = counts.astype(np.float32)

    # Transpose to time-first: (T_total, M_max)
    data_tf = padded.T

    # Run inference in mini-batches
    all_rates = []
    for batch_start in range(0, window_length, batch_size):
        batch_end = min(batch_start + batch_size, window_length)
        batch_inputs = []

        for offset in range(batch_start, batch_end):
            t = start_bin + offset
            # Input window: [t - history_bins, t)
            x = data_tf[t - history_bins:t, :]  # (T, M_max)
            batch_inputs.append(x)

        # Stack and forward pass
        x_batch = torch.tensor(
            np.stack(batch_inputs), dtype=torch.float32,
        ).to(device)

        rates = model(x_batch)  # (batch, M_max)

        # Slice to real neuron channels
        rates_real = rates[:, :m_i].cpu().numpy()
        all_rates.append(rates_real)

    return np.concatenate(all_rates, axis=0)  # (window_length, M_i)


def compute_per_neuron_r(
    real: np.ndarray, pred: np.ndarray,
) -> np.ndarray:
    """
    Compute per-neuron Pearson r between real and predicted data.

    Args:
        real: (T, M_i) ground truth spike counts.
        pred: (T, M_i) predicted rates.

    Returns:
        per_neuron_r: (M_i,) correlation coefficients.
    """
    m_i = real.shape[1]
    per_neuron_r = np.zeros(m_i, dtype=np.float32)
    for j in range(m_i):
        if np.std(real[:, j]) > 0 and np.std(pred[:, j]) > 0:
            per_neuron_r[j] = np.corrcoef(real[:, j], pred[:, j])[0, 1]
    return per_neuron_r


def upload_arrays_to_s3(
    slug: str, npz_dir: Path, summary: dict,
) -> None:
    """
    Upload all .npz files and a summary JSON to S3.

    Args:
        slug: Upload slug (e.g., 'ifer-prediction-arrays').
        npz_dir: Directory containing .npz files.
        summary: Summary dict with per-session metrics.
    """
    from nrp.s3_utils import bucket, retry_with_backoff

    s3_prefix = f"{S3_OUTPUT_PREFIX}/{slug}"

    # Upload summary JSON
    summary_path = npz_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    s3_key = f"{s3_prefix}/summary.json"
    logger.info("Uploading: s3://%s/%s", S3_BUCKET, s3_key)
    retry_with_backoff(lambda: bucket.upload_file(str(summary_path), s3_key))

    # Upload each .npz file
    for npz_file in sorted(npz_dir.glob("session_*.npz")):
        s3_key = f"{s3_prefix}/{npz_file.name}"
        logger.info("Uploading: s3://%s/%s", S3_BUCKET, s3_key)
        retry_with_backoff(
            lambda p=str(npz_file), k=s3_key: bucket.upload_file(p, k),
        )

    logger.info("All files uploaded to s3://%s/%s/", S3_BUCKET, s3_prefix)


def main() -> None:
    """Main pipeline: load model → predict all sessions → upload arrays."""
    args = parse_args()
    seed_everything(args.seed)

    logger.info("=" * 60)
    logger.info("PREDICTION ARRAY GENERATION — IFER Hero Figure")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load configs
    # ------------------------------------------------------------------
    teacher_config = load_config(args.teacher_config)
    data_config = load_config(args.data_config)
    history_bins = data_config.get("history_bins", 10)
    bin_size_ms = data_config.get("bin_size_ms", 50.0)
    train_frac = data_config.get("splits", {}).get("train", 0.7)
    val_frac = data_config.get("splits", {}).get("val", 0.15)

    # ------------------------------------------------------------------
    # 2. Load and preprocess data
    # ------------------------------------------------------------------
    logger.info("Preprocessing multi-session data...")
    cache_dir, multi_meta = preprocess_and_cache(data_config)
    m_max = multi_meta["m_max"]
    num_sessions = multi_meta["num_sessions"]
    logger.info("M_max = %d, %d sessions", m_max, num_sessions)

    # ------------------------------------------------------------------
    # 3. Download and load Mamba checkpoint
    # ------------------------------------------------------------------
    checkpoint_path = download_checkpoint_from_s3(args.teacher_s3_slug)

    logger.info("Creating Mamba model...")
    device = resolve_device()
    model = create_teacher_model(
        config=teacher_config, input_size=m_max,
    )
    ckpt = torch.load(
        str(checkpoint_path), map_location=device, weights_only=True,
    )
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Mamba model loaded: %d params", n_params)

    # ------------------------------------------------------------------
    # 4. Determine sessions to process
    # ------------------------------------------------------------------
    sessions = args.sessions if args.sessions else list(range(num_sessions))
    logger.info("Processing %d sessions", len(sessions))

    # ------------------------------------------------------------------
    # 5. Run inference and save arrays
    # ------------------------------------------------------------------
    output_dir = Path(tempfile.mkdtemp(prefix="ifer_preds_"))
    logger.info("Saving arrays to: %s", output_dir)

    summary = {
        "source_checkpoint": args.teacher_s3_slug,
        "model_params": n_params,
        "history_bins": history_bins,
        "bin_size_ms": bin_size_ms,
        "window_length": args.window_length,
        "num_sessions": len(sessions),
        "sessions": [],
    }

    for sess_idx in sessions:
        logger.info("\n%s", "=" * 60)
        logger.info("Session %03d", sess_idx)
        logger.info("=" * 60)

        # Load session data
        npy_path = cache_dir / f"session_{sess_idx:03d}.npy"
        if not npy_path.exists():
            logger.warning("  Session cache not found: %s — skipping", npy_path)
            continue

        counts = np.load(npy_path).astype(np.float32)  # (M_i, T_total)
        m_i, t_total = counts.shape
        logger.info("  M_i=%d neurons, T_total=%d bins", m_i, t_total)

        # Determine validation window (center of val split)
        train_end = int(t_total * train_frac)
        val_end = int(t_total * (train_frac + val_frac))
        val_mid = (train_end + val_end) // 2
        start_bin = max(
            val_mid - args.window_length // 2,
            train_end + history_bins,
        )
        # Clamp to avoid exceeding val split
        if start_bin + args.window_length > val_end:
            start_bin = val_end - args.window_length
        actual_length = min(args.window_length, val_end - start_bin)

        logger.info(
            "  Val window: bins [%d, %d) (%.1fs)",
            start_bin, start_bin + actual_length,
            actual_length * bin_size_ms / 1000,
        )

        # Ground truth for the window (T, M_i)
        gt = counts[:, start_bin:start_bin + actual_length].T.astype(
            np.float32,
        )

        # Run Mamba inference
        pred = predict_session(
            model=model,
            counts=counts,
            m_max=m_max,
            history_bins=history_bins,
            start_bin=start_bin,
            window_length=actual_length,
            device=device,
        )

        # Compute per-neuron correlations
        per_neuron_r = compute_per_neuron_r(gt, pred)
        mean_r = float(np.mean(per_neuron_r))
        mean_rates = gt.mean(axis=0)
        logger.info("  Mean Pearson r = %.4f", mean_r)

        # Save as .npz
        npz_path = output_dir / f"session_{sess_idx:03d}.npz"
        np.savez_compressed(
            npz_path,
            gt=gt,
            pred=pred,
            per_neuron_r=per_neuron_r,
            mean_rates=mean_rates,
            session_idx=sess_idx,
            m_i=m_i,
            bin_size_ms=bin_size_ms,
            start_bin=start_bin,
            window_length=actual_length,
        )
        size_kb = npz_path.stat().st_size / 1024
        logger.info("  Saved: %s (%.1f KB)", npz_path.name, size_kb)

        # Record in summary
        summary["sessions"].append({
            "session_idx": sess_idx,
            "m_i": m_i,
            "mean_r": mean_r,
            "pct_positive_r": float(np.mean(per_neuron_r > 0) * 100),
            "start_bin": start_bin,
            "window_length": actual_length,
        })

    # ------------------------------------------------------------------
    # 6. Upload to S3
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("UPLOAD TO S3")
    logger.info("=" * 60)

    # Add overall stats to summary
    all_r = [s["mean_r"] for s in summary["sessions"]]
    summary["overall_mean_r"] = float(np.mean(all_r))
    summary["overall_median_r"] = float(np.median(all_r))

    upload_arrays_to_s3(args.upload_slug, output_dir, summary)

    # ------------------------------------------------------------------
    # 7. Print final summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info("  Sessions processed: %d", len(summary["sessions"]))
    logger.info("  Overall mean r:     %.4f", summary["overall_mean_r"])
    logger.info("  Overall median r:   %.4f", summary["overall_median_r"])
    logger.info("  Upload slug:        %s", args.upload_slug)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
