"""
Compute population-level metrics for any architecture (teacher or SNN student).

Generalizes nrp_teacher_pop_metrics.py to also handle StudentSNN models.
Downloads checkpoint from S3, runs inference on val split, computes:
  - pearson_r (per-neuron, weighted)
  - population_rate_r
  - spatial_pattern_r
  - population_cosine_sim

Usage (NRP):
    # Teacher (Mamba/Transformer/LRU/LSTM)
    python scripts/nrp_pop_metrics_general.py \
        --model-type teacher \
        --model-config configs/teacher/nrp_teacher_mamba.yaml \
        --checkpoint-slug 2026-03-26_baseline-mamba-v12 \
        --s3-cache-prefix jrm/spike-prophecy/inputs/steinmetz-session-cache \
        --upload-slug pop-metrics-mamba-steinmetz

    # Student SNN (2L standalone)
    python scripts/nrp_pop_metrics_general.py \
        --model-type student \
        --model-config configs/student/standalone_snn.yaml \
        --checkpoint-slug snn-standalone-v12b \
        --s3-cache-prefix jrm/spike-prophecy/inputs/steinmetz-session-cache \
        --upload-slug pop-metrics-snn-2l-steinmetz

    # LSTM teacher
    python scripts/nrp_pop_metrics_general.py \
        --model-type teacher \
        --model-config configs/archive/teacher/nrp_teacher.yaml \
        --checkpoint-slug 2026-02-21_lstm-lr5e4 \
        --s3-cache-prefix jrm/spike-prophecy/inputs/steinmetz-session-cache \
        --upload-slug pop-metrics-lstm-steinmetz
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
logger = logging.getLogger("nrp_pop_metrics_general")

# ---------------------------------------------------------------------------
# S3 Configuration
# ---------------------------------------------------------------------------
S3_BUCKET = "braingeneersdev"
S3_OUTPUT_PREFIX = "jrm/spike-prophecy/outputs"


def get_s3():
    """Create S3 client for NRP (internal) or local (external)."""
    # Job YAML sets ENDPOINT; some scripts use S3_ENDPOINT — check both
    endpoint = os.environ.get(
        "S3_ENDPOINT",
        os.environ.get("ENDPOINT", "https://s3-west.nrp-nautilus.io"),
    )
    return boto3.client("s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=BotoConfig(retries={"max_attempts": 3}))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute population metrics for any architecture.",
    )
    parser.add_argument(
        "--model-type", type=str, required=True,
        choices=["teacher", "student"],
        help="Model type: 'teacher' (Mamba/Transformer/LRU/LSTM) "
             "or 'student' (StudentSNN).",
    )
    parser.add_argument(
        "--model-config", type=str, required=True,
        help="Path to model config YAML.",
    )
    parser.add_argument(
        "--checkpoint-slug", type=str, required=True,
        help="S3 slug for the model checkpoint.",
    )
    parser.add_argument(
        "--s3-cache-prefix", type=str, required=True,
        help="S3 prefix for pre-cached session .npy files.",
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
    """Download pre-cached session files from S3."""
    local_cache_dir = Path(local_cache_dir)
    local_cache_dir.mkdir(parents=True, exist_ok=True)

    # Download metadata.json
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


def load_model(model_type, config_path, checkpoint_path, m_max, device):
    """Load either a teacher or student model from config + checkpoint.

    Args:
        model_type: 'teacher' or 'student'.
        config_path: Path to the model config YAML.
        checkpoint_path: Local path to the checkpoint file.
        m_max: Maximum neuron count (input/output size).
        device: Torch device.

    Returns:
        Loaded model in eval mode.
    """
    config = load_config(config_path)

    if model_type == "teacher":
        # Use the standard teacher model factory
        from src.models.common import create_teacher_model
        model = create_teacher_model(config, m_max)
    else:
        # Load StudentSNN
        from src.models.student import StudentSNN
        model_cfg = config.get("model", config.get("student", {}))
        model = StudentSNN(
            input_size=m_max,
            hidden_size=model_cfg.get("hidden_size", 256),
            beta=model_cfg.get("beta", 0.9),
            threshold=model_cfg.get("threshold", 1.0),
            output_size=m_max,
            gradient_slope=model_cfg.get("gradient_slope", 25.0),
            learn_beta=model_cfg.get("learn_beta", True),
            num_layers=model_cfg.get("num_layers", 2),
            neuron_type=model_cfg.get("neuron_type", "rsynaptic"),
            alpha=model_cfg.get("alpha", 0.85),
            use_layer_norm=model_cfg.get("use_layer_norm", False),
            dropout=model_cfg.get("dropout", 0.0),
            learn_threshold=model_cfg.get("learn_threshold", False),
            readout_mode=model_cfg.get("readout_mode", "mean"),
            auxiliary_heads=model_cfg.get("auxiliary_heads", None),
            sgc_enabled=model_cfg.get("sgc_enabled", False),
        )

    model = model.to(device)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Try multiple checkpoint key formats
    if isinstance(ckpt, dict):
        sd = ckpt.get("model_state_dict",
             ckpt.get("student_state_dict", ckpt))
    else:
        sd = ckpt

    # Load with strict=False to handle minor mismatches
    result = model.load_state_dict(sd, strict=False)
    if result.missing_keys:
        logger.warning("Missing keys: %s", result.missing_keys[:5])
    if result.unexpected_keys:
        logger.warning("Unexpected keys: %s", result.unexpected_keys[:5])

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded: %s params, type=%s", f"{n_params:,}", model_type)
    return model, n_params


@torch.no_grad()
def eval_session(model, counts, m_max, device, history=10,
                 train_frac=0.7, val_frac=0.15, session_id=None):
    """Run inference on val split of a single session.

    Args:
        model: Teacher or student model.
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

        # Forward pass — handle session_id for session-head models
        try:
            y_hat = model(x, session_id=session_id)
        except TypeError:
            y_hat = model(x)

        # Handle tuple/dict outputs
        if isinstance(y_hat, tuple):
            y_hat = y_hat[0]
        elif isinstance(y_hat, dict):
            y_hat = y_hat.get("rates", y_hat.get("output", list(y_hat.values())[0]))

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
    """Main entrypoint."""
    args = parse_args()
    seed_everything(42)
    device = resolve_device()
    logger.info("Device: %s", device)
    s3 = get_s3()

    # Download session cache from S3
    local_cache = Path("/tmp/session_cache")
    cache_dir, metadata = download_session_cache(
        s3, args.s3_cache_prefix, local_cache
    )
    m_max = metadata["m_max"]
    sessions = metadata["sessions"]
    n_sessions = len(sessions)
    logger.info("Dataset: %d sessions, m_max=%d", n_sessions, m_max)

    # Download and load checkpoint
    ckpt_key = f"{S3_OUTPUT_PREFIX}/{args.checkpoint_slug}/{args.checkpoint_file}"
    local_ckpt = f"/tmp/{args.checkpoint_slug}_{args.checkpoint_file}"
    if not os.path.exists(local_ckpt):
        logger.info("Downloading checkpoint: %s", ckpt_key)
        s3.download_file(S3_BUCKET, ckpt_key, local_ckpt)

    # Load model
    model, n_params = load_model(
        args.model_type, args.model_config, local_ckpt, m_max, device
    )

    # Evaluate all sessions
    t0 = time.time()
    session_results = []
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
        "model_type": args.model_type,
        "model_config": args.model_config,
        "checkpoint_slug": args.checkpoint_slug,
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
