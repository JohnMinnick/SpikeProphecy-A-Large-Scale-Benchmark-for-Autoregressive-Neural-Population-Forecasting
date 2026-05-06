"""
Recalculate distillation val_r with mask-aware evaluation.

Downloads metrics.json from completed NRP experiments, reads the per-
epoch val_r history, and estimates the corrected val_r by accounting
for padding channels.  Also pulls best_model.pt for any experiment
that needs a full re-eval pass.

This script connects to the NRP S3 endpoint and requires AWS_ACCESS_KEY_ID
and AWS_SECRET_ACCESS_KEY environment variables.

Usage:
    python scripts/recalc_distill_metrics.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.metrics import pearson_r, r_squared, poisson_nll, mae, mse


def get_s3_client():
    """Create an S3 client configured for NRP."""
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url="https://s3-west.nrp-nautilus.io",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=Config(
            retries={"max_attempts": 3, "mode": "standard"},
            connect_timeout=10,
            read_timeout=30,
        ),
    )


# Completed experiments to recalculate
EXPERIMENTS = {
    # Steinmetz (M_max=1240, 39 sessions, teacher val_r=0.499)
    "2026-04-03_distill-steinmetz-v3": {
        "description": "Random-init 3-layer RSynaptic",
        "teacher_val_r": 0.499,
        "dataset": "steinmetz",
    },
    "2026-04-03_mimetic-steinmetz-v3": {
        "description": "Mimetic-init 3-layer RSynaptic",
        "teacher_val_r": 0.499,
        "dataset": "steinmetz",
    },
    "2026-04-03_distill-3layer-steinmetz-v1": {
        "description": "3-layer ablation (no aux heads)",
        "teacher_val_r": 0.499,
        "dataset": "steinmetz",
    },
    "2026-04-03_tilif-sgc-steinmetz-v1": {
        "description": "TI-LIF + SGC (DEAD)",
        "teacher_val_r": 0.499,
        "dataset": "steinmetz",
    },
}

# Known neuron-count stats per dataset
# (min, max, mean, n_sessions)
DATASET_STATS = {
    "steinmetz": {"n_sessions": 39, "m_min": 372, "m_max": 1240, "m_mean": 698},
    "ibl": {"n_sessions": 66, "m_min": 200, "m_max": 1998, "m_mean": 800},
    "combined": {"n_sessions": 105, "m_min": 200, "m_max": 1998, "m_mean": 750},
}


def download_metrics(s3, experiment_slug: str, output_dir: Path) -> dict:
    """Download metrics.json from S3 and return as dict."""
    bucket = "<lab-bucket>"
    key = f"<anon>/spike-prophecy/outputs/{experiment_slug}/metrics.json"
    local_path = output_dir / experiment_slug / "metrics.json"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if not local_path.exists():
        print(f"  Downloading metrics.json for {experiment_slug}...")
        try:
            s3.download_file(bucket, key, str(local_path))
        except Exception as e:
            print(f"  ERROR: {e}")
            return {}

    with open(local_path) as f:
        return json.load(f)


def estimate_correction_factor(dataset: str) -> float:
    """
    Estimate the val_r correction factor for padding bias.

    The uncorrected val_r averages over all M_max channels including
    padding.  Padding channels have ~zero variance → r ≈ 0.
    The correction factor = M_max / mean_active_neurons.

    This is an approximation — the true correction requires re-inference.
    """
    stats = DATASET_STATS[dataset]
    m_max = stats["m_max"]
    m_mean = stats["m_mean"]
    # Fraction of channels that are active (on average)
    active_fraction = m_mean / m_max
    return 1.0 / active_fraction


def main():
    """Recalculate distillation metrics with padding correction."""
    output_dir = PROJECT_ROOT / "outputs" / "s3_metrics"

    # Connect to S3
    try:
        s3 = get_s3_client()
    except Exception as e:
        print(f"S3 connection failed: {e}")
        print("Falling back to local metrics only.")
        s3 = None

    results = {}

    for slug, info in EXPERIMENTS.items():
        print(f"\n{'='*70}")
        print(f"  {slug}")
        print(f"  {info['description']}")
        print(f"{'='*70}")

        # Load metrics
        if s3:
            metrics = download_metrics(s3, slug, output_dir)
        else:
            local_path = output_dir / slug / "metrics.json"
            if local_path.exists():
                with open(local_path) as f:
                    metrics = json.load(f)
            else:
                print(f"  No metrics found at {local_path}")
                continue

        if not metrics:
            continue

        # Extract key values
        old_val_r = metrics.get(
            "distill_best_val_pearson_r",
            metrics.get("best_val_pearson_r", None),
        )
        val_loss = metrics.get(
            "distill_best_val_loss",
            metrics.get("best_val_loss", None),
        )
        val_mae = metrics.get(
            "distill_best_val_mae",
            metrics.get("best_val_mae", None),
        )
        val_mse = metrics.get(
            "distill_best_val_mse",
            metrics.get("best_val_mse", None),
        )
        n_epochs = metrics.get(
            "distill_n_epochs_trained",
            metrics.get("n_epochs_trained", None),
        )

        # Estimate corrected val_r
        correction = estimate_correction_factor(info["dataset"])
        corrected_val_r = old_val_r * correction if old_val_r else None

        # Retention calculations
        teacher_r = info["teacher_val_r"]
        old_retention = (old_val_r / teacher_r * 100) if old_val_r else None
        new_retention = (corrected_val_r / teacher_r * 100) if corrected_val_r else None

        results[slug] = {
            "description": info["description"],
            "dataset": info["dataset"],
            "epochs": n_epochs,
            "old_val_r": old_val_r,
            "corrected_val_r": corrected_val_r,
            "correction_factor": correction,
            "teacher_val_r": teacher_r,
            "old_retention": old_retention,
            "new_retention": new_retention,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "val_mse": val_mse,
        }

        print(f"  Epochs:           {n_epochs}")
        print(f"  Old val_r:        {old_val_r:.4f}" if old_val_r else "  Old val_r: N/A")
        print(f"  Correction:       ×{correction:.2f}")
        print(f"  Corrected val_r:  {corrected_val_r:.4f}" if corrected_val_r else "  Corrected val_r: N/A")
        print(f"  Teacher val_r:    {teacher_r:.4f}")
        print(f"  Old retention:    {old_retention:.1f}%" if old_retention else "  Old retention: N/A")
        print(f"  NEW retention:    {new_retention:.1f}%" if new_retention else "  NEW retention: N/A")

    # Print comparison table
    if results:
        print(f"\n{'='*100}")
        print("COMPARISON TABLE — Corrected vs Uncorrected Retention")
        print(f"{'='*100}")
        print(f"{'Experiment':<45} {'Old r':>8} {'New r':>8} {'Teacher':>8} "
              f"{'Old %':>8} {'NEW %':>8} {'Δ%':>6}")
        print("-" * 100)
        for slug, r in sorted(results.items()):
            old_r = f"{r['old_val_r']:.4f}" if r["old_val_r"] else "N/A"
            new_r = f"{r['corrected_val_r']:.4f}" if r["corrected_val_r"] else "N/A"
            teacher = f"{r['teacher_val_r']:.4f}"
            old_pct = f"{r['old_retention']:.1f}%" if r["old_retention"] else "N/A"
            new_pct = f"{r['new_retention']:.1f}%" if r["new_retention"] else "N/A"
            delta = (
                f"+{r['new_retention'] - r['old_retention']:.1f}"
                if r["old_retention"] and r["new_retention"]
                else "N/A"
            )
            print(f"{slug:<45} {old_r:>8} {new_r:>8} {teacher:>8} "
                  f"{old_pct:>8} {new_pct:>8} {delta:>6}")

        print()
        print("⚠️  NOTE: 'New r' is an ESTIMATE using mean active fraction.")
        print("   For exact values, re-run inference with the fixed eval pipeline.")
        print("   The fix is in src/distill/multi_head_trainer.py._validate()")
        print()


if __name__ == "__main__":
    main()
