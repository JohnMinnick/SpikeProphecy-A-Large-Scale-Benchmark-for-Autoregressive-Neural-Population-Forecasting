"""
Quick eval script: compute proper per-session Pearson r for session-head models.

Downloads checkpoint from S3 (or uses local), loads data per-session,
runs inference with the correct session head, computes per-session r
from full sufficient statistics, and reports the weighted average.

Usage (NRP or local):
    python scripts/eval_session_heads.py \
        --teacher-config configs/teacher/nrp_teacher_transformer_session_heads.yaml \
        --data-config configs/data/steinmetz_multi_nrp_50ms_no_cov.yaml \
        --checkpoint outputs/some_checkpoint/best_model.pt

    # From S3:
    python scripts/eval_session_heads.py \
        --teacher-config configs/teacher/nrp_teacher_transformer_session_heads.yaml \
        --data-config configs/data/steinmetz_multi_nrp_50ms_no_cov.yaml \
        --checkpoint-slug 2026-03-16_transformer-session-heads
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "nrp"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_session_heads")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Compute proper per-session Pearson r for session-head models.",
    )
    p.add_argument("--teacher-config", type=str, required=True)
    p.add_argument("--data-config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Local path to model checkpoint.")
    p.add_argument("--checkpoint-slug", type=str, default=None,
                   help="S3 slug to download checkpoint from.")
    p.add_argument("--checkpoint-file", type=str, default="best_model.pt")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--output", type=str, default=None,
                   help="Path to save results JSON.")
    return p.parse_args()


def compute_per_session_r(
    model, cache_dir, data_config, multi_meta, device, batch_size=256,
):
    """
    Compute per-session Pearson r using full sufficient statistics.

    For each session:
      1. Load that session's data
      2. Run inference with the correct session head
      3. Accumulate per-channel sufficient statistics across all batches
      4. Compute per-channel r, then weighted average for the session

    Returns:
        List of per-session result dicts and aggregate stats.
    """
    from src.data.multi_session_loader import MaskedSpikeCountDataset
    from torch.utils.data import DataLoader
    from pathlib import Path

    n_sessions = multi_meta.get("num_sessions", len(multi_meta["sessions"]))
    m_max = multi_meta["m_max"]
    history_bins = data_config.get("history_bins", 10)
    splits = data_config.get("splits", {"train": 0.7, "val": 0.15, "test": 0.15})
    session_results = []

    for sid in range(n_sessions):
        sess_info = multi_meta["sessions"][sid]
        m_i = sess_info["num_units"]
        t_total = sess_info["num_bins"]

        # Load cached count matrix (uint8) -> float32, pad to m_max
        npy_path = Path(cache_dir) / f"session_{sid:03d}.npy"
        counts_u8 = np.load(npy_path)  # (m_i, t_i)
        counts = counts_u8.astype(np.float32)

        # Pad to m_max along neuron axis
        if m_i < m_max:
            pad = np.zeros((m_max - m_i, t_total), dtype=np.float32)
            counts_padded = np.concatenate([counts, pad], axis=0)
        else:
            counts_padded = counts

        # Slice to val split
        train_end = int(t_total * splits["train"])
        val_end = train_end + int(t_total * splits["val"])
        val_counts = counts_padded[:, train_end:val_end]
        val_len = val_end - train_end

        if val_len <= history_bins:
            logger.warning("Session %d: val split too short (%d bins) - skipping", sid, val_len)
            continue

        # Build single-session mask: 1 for real channels, 0 for padding
        session_mask = np.zeros((1, m_max), dtype=np.float32)
        session_mask[0, :m_i] = 1.0

        # mask_index: all bins belong to session 0 (single-session dataset)
        mask_index = np.zeros(val_len, dtype=np.int32)

        # Build dataset
        ds = MaskedSpikeCountDataset(
            spike_counts=val_counts,
            mask_index=mask_index,
            session_masks=session_mask,
            history_bins=history_bins,
        )

        if len(ds) == 0:
            logger.warning("Session %d: no val samples - skipping", sid)
            continue

        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        # Per-channel sufficient statistics (float64 for precision)
        sum_x = torch.zeros(m_i, dtype=torch.float64, device=device)
        sum_y = torch.zeros(m_i, dtype=torch.float64, device=device)
        sum_xy = torch.zeros(m_i, dtype=torch.float64, device=device)
        sum_x2 = torch.zeros(m_i, dtype=torch.float64, device=device)
        sum_y2 = torch.zeros(m_i, dtype=torch.float64, device=device)
        ch_n = torch.zeros(m_i, dtype=torch.float64, device=device)
        n_samples = 0

        model.eval()
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 3:
                    x, y, mask = batch
                    mask = mask.to(device)
                elif len(batch) == 2:
                    x, y = batch
                    mask = None
                else:
                    x, y, mask, _ = batch
                    mask = mask.to(device)

                x = x.to(device)
                y = y.to(device)

                # Forward with session_id for correct head
                session_id_str = f"session_{sid:03d}"
                y_hat = model(x, session_id=session_id_str)
                y_hat = y_hat.float()

                # Slice targets to match session output dim
                out_dim = y_hat.shape[-1]
                if y.shape[-1] != out_dim:
                    y = y[:, :out_dim]
                    if mask is not None:
                        mask = mask[:, :out_dim]

                y_hat_d = y_hat.double()
                y_d = y.double()
                b = y.shape[0]
                n_samples += b

                if mask is not None:
                    mask_d = mask.double()
                    mask_sum = mask_d.sum(dim=0)
                    sum_x += (y_hat_d * mask_d).sum(dim=0)
                    sum_y += (y_d * mask_d).sum(dim=0)
                    sum_xy += (y_hat_d * y_d * mask_d).sum(dim=0)
                    sum_x2 += (y_hat_d.pow(2) * mask_d).sum(dim=0)
                    sum_y2 += (y_d.pow(2) * mask_d).sum(dim=0)
                    ch_n += mask_sum
                else:
                    sum_x += y_hat_d.sum(dim=0)
                    sum_y += y_d.sum(dim=0)
                    sum_xy += (y_hat_d * y_d).sum(dim=0)
                    sum_x2 += y_hat_d.pow(2).sum(dim=0)
                    sum_y2 += y_d.pow(2).sum(dim=0)
                    ch_n += b

        # Compute per-channel Pearson r from sufficient statistics
        eps = 1e-8
        n = ch_n.clamp(min=1.0)
        cov = n * sum_xy - sum_x * sum_y
        var_x = (n * sum_x2 - sum_x.pow(2)).clamp(min=0)
        var_y = (n * sum_y2 - sum_y.pow(2)).clamp(min=0)
        denom = (var_x * var_y).sqrt().clamp(min=eps)
        per_ch_r = cov / denom

        # Weighted average across channels
        ch_weight = ch_n / ch_n.sum().clamp(min=1.0)
        session_r = float((per_ch_r * ch_weight).sum())

        # Also compute unweighted mean for comparison
        session_r_mean = float(per_ch_r.mean())

        result = {
            "session": sid,
            "m_i": m_i,
            "n_samples": n_samples,
            "pearson_r_weighted": session_r,
            "pearson_r_mean": session_r_mean,
        }
        session_results.append(result)
        logger.info(
            "Session %2d: m_i=%3d, n=%5d, r_weighted=%.4f, r_mean=%.4f",
            sid, m_i, n_samples, session_r, session_r_mean,
        )

    # Aggregate across sessions
    if session_results:
        # Weight by neuron count (m_i)
        total_neurons = sum(r["m_i"] for r in session_results)
        agg_r = sum(r["pearson_r_weighted"] * r["m_i"] for r in session_results) / max(total_neurons, 1)
        # Simple average
        simple_avg_r = np.mean([r["pearson_r_weighted"] for r in session_results])
    else:
        agg_r = 0.0
        simple_avg_r = 0.0

    logger.info("=" * 60)
    logger.info("AGGREGATE: neuron-weighted r = %.4f", agg_r)
    logger.info("AGGREGATE: simple average r  = %.4f", simple_avg_r)
    logger.info("AGGREGATE: n_sessions = %d, total_neurons = %d",
                len(session_results), total_neurons)
    logger.info("=" * 60)

    return {
        "per_session": session_results,
        "aggregate_r_neuron_weighted": agg_r,
        "aggregate_r_simple_avg": simple_avg_r,
        "n_sessions": len(session_results),
        "total_neurons": total_neurons,
    }


def main():
    """Main evaluation entry point."""
    args = parse_args()

    from src.models.teacher import create_teacher_model
    from src.data.multi_session_loader import preprocess_and_cache
    from src.utils.config import load_config
    from src.utils.device import resolve_device

    # Load configs
    teacher_config = load_config(args.teacher_config)
    data_config = load_config(args.data_config)
    device = resolve_device()

    # Download NWB data from S3 if running on NRP (data/raw/ is empty)
    data_dir = PROJECT_ROOT / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    if not list(data_dir.glob("*.nwb")):
        logger.info("No local NWB files — downloading from S3...")
        try:
            from s3_utils import list_files, download_single_file
            s3_prefix = os.environ.get(
                "S3_DATA_PREFIX", "jrm/spike-prophecy/inputs",
            )
            all_keys = list_files(s3_prefix)
            nwb_keys = [k for k in all_keys if k.endswith(".nwb")]
            logger.info("Found %d NWB files on S3", len(nwb_keys))
            for key in nwb_keys:
                filename = os.path.basename(key)
                local_path = str(data_dir / filename)
                logger.info("Downloading %s → %s", key, local_path)
                download_single_file(key=key, local_path=local_path)
        except ImportError:
            logger.warning("s3_utils not available — assuming local data")

    # Preprocess data
    cache_dir, multi_meta = preprocess_and_cache(data_config)
    m_max = multi_meta["m_max"]

    # Build session_dims dict for session-head models
    # Maps "session_XXX" → neuron count for each session
    # metadata['sessions'] is a list of dicts with 'num_units' per session
    session_dims = {
        f"session_{s['index']:03d}": s["num_units"]
        for s in multi_meta["sessions"]
    }
    n_sessions = len(session_dims)
    logger.info("Session dims: %d sessions, neuron counts range %d–%d",
                n_sessions, min(session_dims.values()), max(session_dims.values()))

    # Load model with session-specific output heads
    model = create_teacher_model(
        config=teacher_config, input_size=m_max, session_dims=session_dims,
    )

    # Get checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif args.checkpoint_slug:
        # Download from S3
        from s3_utils import download_single_file
        s3_key = f"jrm/spike-prophecy/outputs/{args.checkpoint_slug}/{args.checkpoint_file}"
        local_dir = PROJECT_ROOT / "checkpoints"
        local_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = str(local_dir / f"{args.checkpoint_slug}_{args.checkpoint_file}")
        if not Path(checkpoint_path).exists():
            logger.info("Downloading: s3://%s -> %s", s3_key, checkpoint_path)
            download_single_file(key=s3_key, local_path=checkpoint_path)
    else:
        raise ValueError("Must specify --checkpoint or --checkpoint-slug")

    # Load weights
    logger.info("Loading checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info("Model loaded: %d params", sum(p.numel() for p in model.parameters()))

    # Run eval
    results = compute_per_session_r(
        model, cache_dir, data_config, multi_meta, device, args.batch_size,
    )

    # Save results
    output_path = args.output or "outputs/session_heads_eval.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved: %s", output_path)


if __name__ == "__main__":
    main()
