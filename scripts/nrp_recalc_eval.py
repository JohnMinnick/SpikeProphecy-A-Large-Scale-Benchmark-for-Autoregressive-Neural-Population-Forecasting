"""
NRP eval-only recalculation of distillation val_r.

Uses the existing SessionCyclingLoader (which handles NWB download
from S3 + caching) and the student SNN to run one validation pass
with corrected per-session Pearson r.

Workflow:
  1. Download NWB files from S3 (via session cycling loader)
  2. Download student checkpoint from S3
  3. Run per-session inference on validation split
  4. Compute per-session metrics (Pearson r, R^2, MAE, NLL)
  5. Compute neuron-weighted average across sessions
  6. Upload results JSON to S3

Usage (NRP):
    python scripts/nrp_recalc_eval.py \
        --checkpoint-slug 2026-04-03_distill-steinmetz-v3 \
        --data-config configs/data/steinmetz_multi_nrp_50ms_no_cov.yaml \
        --student-config configs/student/distill_nrp.yaml \
        --teacher-val-r 0.499

Environment:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_ENDPOINT
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
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.metrics import pearson_r, r_squared, poisson_nll, mae, mse
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.data.multi_session_loader import (
    SessionCyclingLoader,
    MaskedSpikeCountDataset,
    build_channel_mask,
    pad_to_channels,
    preprocess_and_cache,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nrp_recalc")


def download_checkpoint(slug: str, output_dir: Path) -> Path:
    """Download best_model.pt from S3 using NRP s3_utils."""
    nrp_dir = PROJECT_ROOT / "nrp"
    sys.path.insert(0, str(nrp_dir))
    from s3_utils import download_checkpoint as s3_download_ckpt

    s3_key = f"jrm/spike-prophecy/outputs/{slug}/best_model.pt"
    local_path = output_dir / f"{slug}_best_model.pt"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if not local_path.exists():
        logger.info("Downloading checkpoint: %s", s3_key)
        s3_download_ckpt(key=s3_key, local_path=str(local_path))
        logger.info("Checkpoint saved: %s", local_path)

    return local_path


def load_student_model(config_path: str, checkpoint_path: Path,
                       m_max: int, device: torch.device):
    """Load student SNN from config + checkpoint.

    Must construct the model with the SAME parameters as the training
    script (train_distill_multi_head.py) so that all checkpoint weights
    match exactly.
    """
    config = load_config(config_path)
    # Training script reads from config["model"], not config["student"]
    model_cfg = config.get("model", config.get("student", {}))

    from src.models.student import StudentSNN

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

    # Load checkpoint — use strict=True to catch mismatches
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif "student_state_dict" in ckpt:
        sd = ckpt["student_state_dict"]
    else:
        sd = ckpt

    # Log checkpoint info
    if isinstance(ckpt, dict) and "epoch" in ckpt:
        logger.info("Checkpoint from epoch %d", ckpt["epoch"])

    # Load with strict=True first; fall back to strict=False with warning
    try:
        model.load_state_dict(sd, strict=True)
        logger.info("Checkpoint loaded (strict=True, all keys matched)")
    except RuntimeError as e:
        logger.warning("Strict load failed: %s", e)
        logger.warning("Falling back to strict=False — CHECK FOR ISSUES")
        result = model.load_state_dict(sd, strict=False)
        if result.missing_keys:
            logger.warning("Missing keys: %s", result.missing_keys)
        if result.unexpected_keys:
            logger.warning("Unexpected keys: %s", result.unexpected_keys)

    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Loaded student: %d params, %d layers, %s, learn_beta=%s",
        n_params, model_cfg.get("num_layers", 2),
        model_cfg.get("neuron_type", "rsynaptic"),
        model_cfg.get("learn_beta", True),
    )
    return model


@torch.no_grad()
def evaluate_per_session(
    model, loader: SessionCyclingLoader, device: torch.device,
    batch_size: int = 512,
) -> dict:
    """
    Per-session evaluation matching the teacher's Trainer.evaluate().

    For each session:
      1. Load the session's cached .npy data
      2. Use the validation split
      3. Run student inference
      4. Compute Pearson r on REAL neurons only (first m_i channels)

    Then: neuron-weighted average across all sessions.
    """
    model.eval()
    metadata = loader.metadata
    m_max = metadata["m_max"]
    n_sessions = metadata["num_sessions"]
    history_bins = loader.history_bins

    logger.info("Evaluating %d sessions, M_max=%d", n_sessions, m_max)

    session_results = []

    for sess_idx in range(n_sessions):
        npy_path = loader.cache_dir / f"session_{sess_idx:03d}.npy"
        if not npy_path.exists():
            logger.warning("Missing session_%03d.npy, skipping", sess_idx)
            continue

        sess_info = metadata["sessions"][sess_idx]
        m_i = sess_info["num_units"]

        # Load raw counts
        counts_u8 = np.load(npy_path)

        # Get val split slice
        split_start, split_end = loader._get_split_slice(sess_info)
        split_len = split_end - split_start
        if split_len <= history_bins:
            logger.warning("Session %03d: val too short (%d bins)", sess_idx, split_len)
            del counts_u8
            continue

        val_counts = counts_u8[:, split_start:split_end].astype(np.int32)
        del counts_u8

        # Pad to M_max
        padded = pad_to_channels(val_counts, m_max)
        del val_counts

        # Build dataset
        mask_index = np.zeros(padded.shape[1], dtype=np.int32)
        ds = MaskedSpikeCountDataset(
            spike_counts=padded,
            mask_index=mask_index,
            session_masks=build_channel_mask(m_i, m_max).reshape(1, -1),
            history_bins=history_bins,
            output_channels=m_max,
        )
        del padded

        if len(ds) == 0:
            continue

        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        # Run inference for this session
        all_preds = []
        all_targets = []

        for batch in dl:
            # Unpack: (x, y, mask) from MaskedSpikeCountDataset
            x = batch[0].to(device)
            y = batch[1]  # Keep on CPU for accumulation

            output = model(x)
            if isinstance(output, tuple):
                rates = output[0]
            elif isinstance(output, dict):
                rates = output["rates"]
            else:
                rates = output

            # Slice to real neurons only
            all_preds.append(rates[:, :m_i].cpu())
            all_targets.append(y[:, :m_i])

        preds_t = torch.cat(all_preds, dim=0)
        targets_t = torch.cat(all_targets, dim=0)

        # Compute per-session metrics on real neurons only
        s_r = float(pearson_r(preds_t, targets_t))
        s_r2 = float(r_squared(preds_t, targets_t))
        s_mae = float(mae(preds_t, targets_t))
        s_nll = float(poisson_nll(preds_t, targets_t, log_input=False))

        # Per-channel r for diagnostic
        per_ch_r = pearson_r(preds_t, targets_t, per_channel=True)
        r_min = float(per_ch_r.min())
        r_max = float(per_ch_r.max())
        r_median = float(per_ch_r.median())

        session_results.append({
            "session_idx": sess_idx,
            "n_neurons": m_i,
            "n_samples": int(preds_t.shape[0]),
            "pearson_r": s_r,
            "r_squared": s_r2,
            "mae": s_mae,
            "poisson_nll": s_nll,
            "r_min": r_min,
            "r_max": r_max,
            "r_median": r_median,
        })

        logger.info(
            "  Session %03d: m_i=%4d, r=%.4f (min=%.3f med=%.3f max=%.3f), "
            "R2=%.4f, MAE=%.4f",
            sess_idx, m_i, s_r, r_min, r_median, r_max, s_r2, s_mae,
        )

    # ---------------------------------------------------------------
    # Aggregate: neuron-weighted average
    # ---------------------------------------------------------------
    if not session_results:
        return {"error": "No sessions evaluated"}

    total_neurons = sum(s["n_neurons"] for s in session_results)
    weighted_r = sum(
        s["pearson_r"] * s["n_neurons"] for s in session_results
    ) / max(total_neurons, 1)
    weighted_r2 = sum(
        s["r_squared"] * s["n_neurons"] for s in session_results
    ) / max(total_neurons, 1)

    # Simple (unweighted) average for comparison
    simple_r = float(np.mean([s["pearson_r"] for s in session_results]))
    simple_r2 = float(np.mean([s["r_squared"] for s in session_results]))
    mean_mae = float(np.mean([s["mae"] for s in session_results]))
    mean_nll = float(np.mean([s["poisson_nll"] for s in session_results]))

    results = {
        "n_sessions": len(session_results),
        "total_neurons": total_neurons,
        "m_max": m_max,
        "weighted_pearson_r": weighted_r,
        "simple_mean_pearson_r": simple_r,
        "weighted_r_squared": weighted_r2,
        "simple_mean_r_squared": simple_r2,
        "mean_mae": mean_mae,
        "mean_poisson_nll": mean_nll,
        "per_session": session_results,
    }

    return results


def upload_results(results: dict, slug: str, output_path: str):
    """Upload results JSON to S3 using NRP s3_utils."""
    try:
        nrp_dir = PROJECT_ROOT / "nrp"
        sys.path.insert(0, str(nrp_dir))
        from s3_utils import upload_files

        s3_prefix = "jrm/spike-prophecy/outputs"
        upload_files(s3_prefix, slug, output_path)
        logger.info("Uploaded results to S3: %s/%s", s3_prefix, slug)
    except Exception as e:
        logger.warning("S3 upload failed: %s", e)


def main():
    """Main entrypoint for eval-only recalculation."""
    parser = argparse.ArgumentParser(
        description="Recalculate distillation val_r with per-session eval.",
    )
    parser.add_argument(
        "--checkpoint-slug", type=str, required=True,
        help="S3 experiment slug (e.g. 2026-04-03_distill-steinmetz-v3).",
    )
    parser.add_argument(
        "--data-config", type=str, required=True,
        help="Path to data config YAML.",
    )
    parser.add_argument(
        "--student-config", type=str, required=True,
        help="Path to student config YAML.",
    )
    parser.add_argument(
        "--teacher-val-r", type=float, default=0.499,
        help="Teacher val_r for retention calculation.",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Local path to save results JSON.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=512,
        help="Batch size for inference.",
    )
    args = parser.parse_args()

    start_time = time.time()
    device = resolve_device()
    logger.info("Device: %s", device)

    # 1. Load data config and download data from S3
    data_config = load_config(args.data_config)
    source_type = data_config.get("source", {}).get("type", "nwb_multi")

    nrp_dir = PROJECT_ROOT / "nrp"
    sys.path.insert(0, str(nrp_dir))
    from s3_utils import list_files, download_single_file

    if source_type == "ibl":
        # -----------------------------------------------------------------
        # Combined / IBL data: download pre-cached .npy session arrays
        # from S3 (same logic as train_snn_standalone.py).
        # This avoids re-running preprocess_and_cache() which would
        # produce a Steinmetz-only cache with wrong M_max.
        # -----------------------------------------------------------------
        ibl_tag = data_config.get("ibl", {}).get("tag", "repeated_site")
        tag_to_prefix = {
            "repeated_site": "jrm/spike-prophecy/inputs/ibl-repeated-site",
            "combined": "jrm/spike-prophecy/inputs/combined-steinmetz-ibl",
        }
        tag_to_cache = {
            "repeated_site": "ibl_repeated_site_cache",
            "combined": "combined_steinmetz_ibl_cache",
        }
        ibl_s3_prefix = tag_to_prefix.get(
            ibl_tag, f"jrm/spike-prophecy/inputs/{ibl_tag}"
        )
        ibl_cache_name = tag_to_cache.get(ibl_tag, f"{ibl_tag}_cache")
        cache_dir = PROJECT_ROOT / "data" / "processed" / ibl_cache_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Downloading combined/IBL cache from S3: %s -> %s",
            ibl_s3_prefix, cache_dir,
        )
        all_keys = list_files(ibl_s3_prefix)
        cache_keys = [
            k for k in all_keys
            if k.endswith(".npy") or k.endswith(".json")
        ]
        logger.info("Found %d cache files in S3", len(cache_keys))
        for key in cache_keys:
            filename = os.path.basename(key)
            local_path = str(cache_dir / filename)
            if not Path(local_path).exists():
                logger.info("  Downloading %s", filename)
                download_single_file(key=key, local_path=local_path)
            else:
                logger.info("  Already cached: %s", filename)

        # Load metadata from the pre-built cache
        meta_path = cache_dir / "metadata.json"
        with open(meta_path) as mf:
            metadata = json.load(mf)
        m_max = metadata["m_max"]
        logger.info(
            "Combined cache loaded: %d sessions, M_max=%d",
            metadata["num_sessions"], m_max,
        )
    else:
        # -----------------------------------------------------------------
        # Steinmetz NWB data: download .nwb files then preprocess
        # -----------------------------------------------------------------
        logger.info("Downloading NWB files from S3...")
        s3_prefix = os.environ.get(
            "S3_DATA_PREFIX", "jrm/spike-prophecy/inputs"
        )
        data_dir = PROJECT_ROOT / "data" / "raw"
        data_dir.mkdir(parents=True, exist_ok=True)

        all_keys = list_files(s3_prefix)
        nwb_keys = [k for k in all_keys if k.endswith(".nwb")]
        logger.info("Found %d NWB files in S3", len(nwb_keys))
        for key in nwb_keys:
            filename = os.path.basename(key)
            local_path = str(data_dir / filename)
            if not Path(local_path).exists():
                logger.info("  Downloading %s", filename)
                download_single_file(key=key, local_path=local_path)
            else:
                logger.info("  Already cached: %s", filename)

        logger.info("Building data cache from config: %s", args.data_config)
        cache_dir, metadata = preprocess_and_cache(data_config)
        m_max = metadata["m_max"]

    # Create cycling loader for val split (used for split boundaries)
    loader = SessionCyclingLoader(
        cache_dir=cache_dir,
        metadata=metadata,
        split="val",
        config=data_config,
        shuffle_sessions=False,
    )
    logger.info("Data loaded: %d sessions, M_max=%d", metadata["num_sessions"], m_max)

    # 2. Download checkpoint
    ckpt_dir = Path("/workspace/checkpoints") if Path("/workspace").exists() else PROJECT_ROOT / "outputs" / "checkpoints"
    ckpt_path = download_checkpoint(args.checkpoint_slug, ckpt_dir)

    # 3. Load student model
    model = load_student_model(args.student_config, ckpt_path, m_max, device)

    # 4. Run per-session evaluation
    results = evaluate_per_session(model, loader, device, args.batch_size)
    results["checkpoint_slug"] = args.checkpoint_slug
    results["teacher_val_r"] = args.teacher_val_r

    # 5. Compute retention
    if results.get("weighted_pearson_r"):
        wr = results["weighted_pearson_r"]
        sr = results["simple_mean_pearson_r"]
        retention_w = wr / args.teacher_val_r * 100
        retention_s = sr / args.teacher_val_r * 100

        results["weighted_retention_pct"] = retention_w
        results["simple_retention_pct"] = retention_s

    # 6. Print summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("CORRECTED DISTILLATION METRICS (per-session eval)")
    print("=" * 70)
    print(f"  Checkpoint:         {args.checkpoint_slug}")
    print(f"  Sessions evaluated: {results['n_sessions']}")
    print(f"  Total neurons:      {results['total_neurons']}")
    print(f"  M_max:              {results['m_max']}")
    print(f"  ---")
    print(f"  Weighted Pearson r: {results['weighted_pearson_r']:.4f}")
    print(f"  Simple mean r:      {results['simple_mean_pearson_r']:.4f}")
    print(f"  Weighted R-squared: {results['weighted_r_squared']:.6f}")
    print(f"  Simple mean R2:     {results['simple_mean_r_squared']:.6f}")
    print(f"  Mean MAE:           {results['mean_mae']:.4f}")
    print(f"  Mean Poisson NLL:   {results['mean_poisson_nll']:.4f}")
    print(f"  ---")
    print(f"  Teacher val_r:      {args.teacher_val_r:.4f}")
    if results.get("weighted_retention_pct"):
        print(f"  Weighted RETENTION: {results['weighted_retention_pct']:.1f}%")
        print(f"  Simple RETENTION:   {results['simple_retention_pct']:.1f}%")
    print(f"  Elapsed:            {elapsed:.1f}s")
    print("=" * 70)

    # 7. Save results
    output_path = args.output_json or f"/workspace/outputs/recalc_{args.checkpoint_slug}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_path)

    # 8. Upload to S3
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        upload_results(results, args.checkpoint_slug, output_path)


if __name__ == "__main__":
    main()
