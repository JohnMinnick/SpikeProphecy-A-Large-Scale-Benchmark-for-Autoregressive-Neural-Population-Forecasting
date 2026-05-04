"""
Compute population-level metrics for Mamba teacher across all sessions.

Runs on NRP (requires mamba-ssm / Linux). Downloads pre-cached .npy session
arrays from S3, runs teacher inference on the val split, and computes:
  - pearson_r (per-neuron, weighted)
  - population_rate_r
  - spatial_pattern_r
  - population_cosine_sim

Uploads results JSON to S3 under the specified slug.

Usage (NRP):
    # Combined (105 sessions)
    python scripts/nrp_teacher_pop_metrics.py \
        --teacher-s3-slug 2026-04-01_scale-combined-105 \
        --teacher-config configs/teacher/nrp_teacher_mamba.yaml \
        --s3-cache-prefix <anon>/spike-prophecy/inputs/combined-steinmetz-ibl \
        --upload-slug teacher-pop-metrics-combined

    # IBL (66 sessions)
    python scripts/nrp_teacher_pop_metrics.py \
        --teacher-s3-slug 2026-04-01_scale-ibl-only \
        --teacher-config configs/teacher/nrp_teacher_mamba.yaml \
        --s3-cache-prefix <anon>/spike-prophecy/inputs/ibl-repeated-site \
        --upload-slug teacher-pop-metrics-ibl

    # Steinmetz (39 sessions) — requires NWB download first
    python scripts/nrp_teacher_pop_metrics.py \
        --teacher-s3-slug 2026-03-26_baseline-mamba-v12 \
        --teacher-config configs/teacher/nrp_teacher_mamba.yaml \
        --s3-cache-prefix <anon>/spike-prophecy/inputs/steinmetz \
        --upload-slug teacher-pop-metrics-steinmetz
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import boto3
from botocore.config import Config as BotoConfig

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.common import create_teacher_model
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.seed import seed_everything
from src.eval.metrics import (
    pearson_r, population_rate_r, spatial_pattern_r, population_cosine_sim,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nrp_teacher_pop_metrics")

# ---------------------------------------------------------------------------
# S3 Configuration
# ---------------------------------------------------------------------------
S3_BUCKET = "braingeneersdev"
S3_OUTPUT_PREFIX = "<anon>/spike-prophecy/outputs"


def get_s3():
    """Create S3 client for NRP."""
    return boto3.client("s3",
        endpoint_url=os.environ.get("S3_ENDPOINT", "https://s3-west.nrp-nautilus.io"),
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=BotoConfig(retries={"max_attempts": 3}))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute Mamba teacher pop metrics on NRP.",
    )
    parser.add_argument(
        "--teacher-config", type=str, required=True,
        help="Path to teacher model config YAML.",
    )
    parser.add_argument(
        "--teacher-s3-slug", type=str, required=True,
        help="S3 slug for the Mamba checkpoint.",
    )
    parser.add_argument(
        "--s3-cache-prefix", type=str, required=True,
        help="S3 prefix for pre-cached session .npy files "
             "(e.g., <anon>/spike-prophecy/inputs/combined-steinmetz-ibl).",
    )
    parser.add_argument(
        "--upload-slug", type=str, required=True,
        help="S3 slug for uploading results.",
    )
    parser.add_argument(
        "--checkpoint-file", type=str, default="best_model.pt",
        help="Checkpoint filename within the S3 slug.",
    )
    return parser.parse_args()


def download_session_cache(s3, s3_prefix, local_cache_dir):
    """
    Download pre-cached session files (metadata.json + session_XXX.npy)
    from S3 into a local directory.

    Returns:
        Tuple of (cache_dir_path, metadata_dict)
    """
    local_cache_dir = Path(local_cache_dir)
    local_cache_dir.mkdir(parents=True, exist_ok=True)

    # Download metadata.json first
    meta_key = f"{s3_prefix}/metadata.json"
    meta_local = local_cache_dir / "metadata.json"
    if not meta_local.exists():
        logger.info("Downloading metadata: %s", meta_key)
        s3.download_file(S3_BUCKET, meta_key, str(meta_local))

    metadata = json.load(open(meta_local))
    n_sessions = len(metadata["sessions"])
    logger.info("Cache has %d sessions, m_max=%d", n_sessions, metadata["m_max"])

    # Download each session .npy file
    for i in range(n_sessions):
        fname = f"session_{i:03d}.npy"
        local_path = local_cache_dir / fname
        if local_path.exists():
            continue
        s3_key = f"{s3_prefix}/{fname}"
        logger.info("  Downloading %s", fname)
        s3.download_file(S3_BUCKET, s3_key, str(local_path))

    logger.info("All %d session files downloaded", n_sessions)
    return local_cache_dir, metadata


@torch.no_grad()
def eval_session(model, counts, m_max, device, history=10,
                 train_frac=0.7, val_frac=0.15, session_id=None):
    """
    Run teacher inference on val split of a single session.

    Args:
        model: Teacher model.
        counts: Raw spike counts, shape (m_i, T_total).
        m_max: Maximum neuron count for padding.
        device: Torch device.
        history: Number of history bins.
        train_frac: Training split fraction.
        val_frac: Validation split fraction.
        session_id: Optional session ID for session-head models.

    Returns:
        Tuple of (metrics_dict, neuron_count), or (None, 0) on failure.
    """
    m_i = counts.shape[0]
    t_total = counts.shape[1]

    # Pad to m_max
    padded = np.zeros((m_max, t_total), dtype=np.float32)
    padded[:m_i] = counts
    data_t = padded.T  # (T_total, m_max)

    # Val window
    train_end = int(t_total * train_frac)
    val_end = int(t_total * (train_frac + val_frac))
    window = val_end - train_end - history
    if window <= 0:
        return None, 0
    start = train_end + history

    # GT: (T, m_i)
    gt = counts[:m_i, start:start + window].T

    # Inference in batches
    batch_size = 256
    all_preds = []
    for bs in range(0, window, batch_size):
        be = min(bs + batch_size, window)
        batch = []
        for o in range(bs, be):
            t = start + o
            batch.append(data_t[t - history:t])
        x = torch.tensor(np.stack(batch), dtype=torch.float32).to(device)

        # Forward pass — teacher may accept session_id kwarg
        try:
            y_hat = model(x, session_id=session_id)
        except TypeError:
            y_hat = model(x)

        if isinstance(y_hat, tuple):
            y_hat = y_hat[0]
        all_preds.append(y_hat[:, :m_i].cpu())

    pred_t = torch.cat(all_preds, dim=0)
    gt_t = torch.tensor(gt, dtype=torch.float32)

    # Compute population metrics
    metrics = {
        "pearson_r": float(pearson_r(pred_t, gt_t)),
        "pop_rate_r": float(population_rate_r(pred_t, gt_t)),
        "spatial_r": float(spatial_pattern_r(pred_t, gt_t)),
        "cosine_sim": float(population_cosine_sim(pred_t, gt_t)),
    }
    return metrics, m_i


def main():
    args = parse_args()
    seed_everything(42)
    device = resolve_device()
    logger.info("Device: %s", device)
    s3 = get_s3()

    # Load teacher config
    teacher_cfg = load_config(args.teacher_config)

    # Download session cache from S3
    local_cache = Path("/tmp/session_cache")
    cache_dir, metadata = download_session_cache(
        s3, args.s3_cache_prefix, local_cache
    )
    m_max = metadata["m_max"]
    sessions = metadata["sessions"]
    n_sessions = len(sessions)
    logger.info("Dataset: %d sessions, m_max=%d", n_sessions, m_max)

    # Build teacher model
    logger.info("Building teacher model...")
    model = create_teacher_model(teacher_cfg, m_max)
    model = model.to(device)

    # Download and load checkpoint
    ckpt_key = f"{S3_OUTPUT_PREFIX}/{args.teacher_s3_slug}/{args.checkpoint_file}"
    local_ckpt = f"/tmp/{args.teacher_s3_slug}_{args.checkpoint_file}"
    if not os.path.exists(local_ckpt):
        logger.info("Downloading checkpoint: %s", ckpt_key)
        s3.download_file(S3_BUCKET, ckpt_key, local_ckpt)
    ckpt = torch.load(local_ckpt, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Teacher model: %s params", f"{n_params:,}")

    # Evaluate all sessions
    t0 = time.time()
    session_results = []  # List of (metrics, n_neurons)
    per_session_details = []

    for i in range(n_sessions):
        sess_file = cache_dir / f"session_{i:03d}.npy"
        if not sess_file.exists():
            logger.warning("Session %d not found, skipping", i)
            continue

        counts = np.load(sess_file).astype(np.float32)
        sess_id = sessions[i].get("session_id",
                  sessions[i].get("file", f"session_{i:03d}"))

        metrics, m_i = eval_session(
            model, counts, m_max, device,
            history=10,
            session_id=sess_id,
        )

        if metrics is not None:
            session_results.append((metrics, m_i))
            per_session_details.append({
                "session_idx": i,
                "session_id": sess_id,
                "num_neurons": m_i,
                **{k: round(v, 4) for k, v in metrics.items()},
            })

        # Progress logging every 10 sessions
        if (i + 1) % 10 == 0 or i == n_sessions - 1:
            elapsed = time.time() - t0
            keys = ["pearson_r", "pop_rate_r", "spatial_r", "cosine_sim"]
            total_n = sum(n for _, n in session_results)
            avg = {}
            for k in keys:
                avg[k] = sum(m[k] * n for m, n in session_results) / max(total_n, 1)
            logger.info(
                "[%3d/%d] %.0fs | r=%.4f | pop_r=%.4f | spat=%.4f | cos=%.4f",
                i + 1, n_sessions, elapsed,
                avg["pearson_r"], avg["pop_rate_r"],
                avg["spatial_r"], avg["cosine_sim"],
            )

    # Final weighted averages
    keys = ["pearson_r", "pop_rate_r", "spatial_r", "cosine_sim"]
    total_n = sum(n for _, n in session_results)
    weighted_avg = {}
    for k in keys:
        weighted_avg[k] = round(
            sum(m[k] * n for m, n in session_results) / max(total_n, 1), 4
        )

    elapsed = time.time() - t0
    logger.info("FINAL (%d sessions, %d neurons, %.0fs): %s",
                len(session_results), total_n, elapsed, weighted_avg)

    # Build results payload
    results = {
        "teacher_slug": args.teacher_s3_slug,
        "s3_cache_prefix": args.s3_cache_prefix,
        "n_sessions": len(session_results),
        "total_neurons": total_n,
        "n_params": n_params,
        "eval_time_s": round(elapsed, 1),
        "weighted_avg": weighted_avg,
        "per_session": per_session_details,
    }

    # Upload to S3
    key = f"{S3_OUTPUT_PREFIX}/{args.upload_slug}/pop_metrics.json"
    body = json.dumps(results, indent=2, default=str)
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=body)
    logger.info("Uploaded results to s3://%s/%s", S3_BUCKET, key)

    # Also save locally
    local_path = Path("/workspace/outputs") / args.upload_slug / "pop_metrics.json"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved local: %s", local_path)


if __name__ == "__main__":
    main()
