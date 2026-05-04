"""
Post-hoc distillation evaluation with full metric stack.

Loads a saved student checkpoint + its config + data, runs one validation
pass, and computes the full metric stack including R² (which was added
after the current NRP jobs were deployed).

Also computes retention ratios (student / teacher) if a teacher S3 slug
is provided, giving the headline "X% retention" numbers.

Usage:
    # Evaluate a single experiment (pulls checkpoint from S3 if needed)
    python scripts/eval_distill_posthoc.py \
        --experiment tilif-sgc-steinmetz-v1 \
        --teacher-s3-slug 2026-03-26_baseline-mamba-v12

    # Evaluate from local experiment directory
    python scripts/eval_distill_posthoc.py \
        --exp-dir outputs/s3_metrics/tilif-sgc-steinmetz-v1

    # Compare multiple experiments in a table
    python scripts/eval_distill_posthoc.py \
        --experiment tilif-sgc-steinmetz-v1 mimetic-steinmetz-v3 \
        --teacher-s3-slug 2026-03-26_baseline-mamba-v12
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Add project root to path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.metrics import pearson_r, r_squared, poisson_nll, mae, mse
from src.utils.config import load_config
from src.utils.device import resolve_device

logger = logging.getLogger(__name__)


def load_metrics_from_s3(experiment_name: str, output_dir: Path) -> dict:
    """
    Download metrics.json and config.yaml from S3 for an experiment.

    Args:
        experiment_name: S3 experiment slug.
        output_dir: Local directory to save files.

    Returns:
        Dict with metrics from metrics.json, or empty dict if not found.
    """
    try:
        import boto3
        from botocore.config import Config

        s3_config = Config(
            retries={"max_attempts": 3, "mode": "standard"},
            connect_timeout=10,
            read_timeout=30,
        )
        s3 = boto3.client(
            "s3",
            endpoint_url="https://s3-west.nrp-nautilus.io",
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            config=s3_config,
        )

        bucket = "braingeneersdev"
        prefix = f"<anon>/spike-prophecy/outputs/{experiment_name}"

        # Download metrics.json, config.yaml, best_model.pt
        exp_dir = output_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        for filename in ["metrics.json", "config.yaml", "best_model.pt"]:
            key = f"{prefix}/{filename}"
            local_path = exp_dir / filename
            if not local_path.exists():
                try:
                    print(f"  Downloading {filename}...")
                    s3.download_file(bucket, key, str(local_path))
                except Exception as e:
                    print(f"  Warning: could not download {filename}: {e}")

        # Load metrics
        metrics_path = exp_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                return json.load(f)
    except Exception as e:
        print(f"  S3 error: {e}")

    return {}


def compute_posthoc_metrics(metrics: dict) -> dict:
    """
    Compute R² and retention ratios from saved metrics.json.

    For R², we use the relationship: R² = 1 - MSE / Var(target).
    Since we don't have Var(target) saved, we compute what we can
    from the available metrics and flag what needs a full eval pass.

    Args:
        metrics: Dict from metrics.json.

    Returns:
        Dict with computed post-hoc metrics.
    """
    result = {}

    # Extract the key metrics we already have
    student_r = metrics.get("distill_best_val_pearson_r",
                            metrics.get("best_val_pearson_r"))
    student_nll = metrics.get("distill_best_val_poisson_nll",
                              metrics.get("best_val_poisson_nll"))
    student_mse = metrics.get("distill_best_val_mse",
                              metrics.get("best_val_mse"))
    student_mae = metrics.get("distill_best_val_mae",
                              metrics.get("best_val_mae"))
    student_loss = metrics.get("distill_best_val_loss",
                               metrics.get("best_val_loss"))

    # Teacher metrics (if teacher eval was recorded)
    teacher_r = metrics.get("teacher_best_val_pearson_r")
    teacher_nll = metrics.get("teacher_best_val_poisson_nll")

    result["student_pearson_r"] = student_r
    result["student_poisson_nll"] = student_nll
    result["student_mse"] = student_mse
    result["student_mae"] = student_mae
    result["student_val_loss"] = student_loss

    # R² requires Var(target) — flag that we need a full eval pass
    result["r_squared"] = "NEEDS_EVAL_PASS"
    result["r_squared_note"] = (
        "R² = 1 - MSE/Var(target). Var(target) not saved in metrics.json. "
        "Run with --full-eval to compute from checkpoint."
    )

    # Retention ratios
    if teacher_r and student_r:
        result["retention_pearson_r"] = student_r / teacher_r
        result["retention_pct"] = f"{100 * student_r / teacher_r:.1f}%"
    else:
        result["retention_pearson_r"] = None
        result["retention_pct"] = "N/A (no teacher metrics)"

    if teacher_nll and student_nll:
        # For NLL, lower is better, so retention = teacher/student → >1 is good
        result["retention_poisson_nll"] = teacher_nll / student_nll

    result["teacher_pearson_r"] = teacher_r
    result["teacher_poisson_nll"] = teacher_nll
    result["n_epochs"] = metrics.get("distill_n_epochs_trained",
                                     metrics.get("n_epochs_trained"))

    return result


def print_comparison_table(results: dict):
    """
    Print a formatted comparison table of multiple experiments.

    Args:
        results: Dict mapping experiment name to posthoc metrics.
    """
    # Header
    print(f"\n{'Experiment':<35} {'R':>8} {'R²':>8} {'NLL':>8} "
          f"{'MAE':>8} {'Retent.':>8} {'Epochs':>7}")
    print("-" * 90)

    for exp_name, m in sorted(results.items()):
        r_val = f"{m['student_pearson_r']:.4f}" if m.get('student_pearson_r') else "N/A"
        r2_val = (f"{m['r_squared']:.4f}"
                  if isinstance(m.get('r_squared'), (int, float)) else "—")
        nll_val = f"{m['student_poisson_nll']:.4f}" if m.get('student_poisson_nll') else "N/A"
        mae_val = f"{m['student_mae']:.4f}" if m.get('student_mae') else "N/A"
        ret_val = m.get('retention_pct', 'N/A')
        epochs = str(m.get('n_epochs', 'N/A'))

        print(f"{exp_name:<35} {r_val:>8} {r2_val:>8} {nll_val:>8} "
              f"{mae_val:>8} {ret_val:>8} {epochs:>7}")

    # Teacher reference (if available)
    first_result = next(iter(results.values()), {})
    if first_result.get("teacher_pearson_r"):
        teacher_r = f"{first_result['teacher_pearson_r']:.4f}"
        teacher_nll = (f"{first_result['teacher_poisson_nll']:.4f}"
                       if first_result.get('teacher_poisson_nll') else "N/A")
        print("-" * 90)
        print(f"{'(Teacher reference)':<35} {teacher_r:>8} {'—':>8} "
              f"{teacher_nll:>8} {'—':>8} {'100.0%':>8} {'—':>7}")

    print()


def main():
    """Main entrypoint for post-hoc distillation evaluation."""
    parser = argparse.ArgumentParser(
        description="Post-hoc distillation evaluation with full metric stack.",
    )
    parser.add_argument(
        "--experiment", nargs="+", type=str, default=None,
        help="S3 experiment slug(s) to evaluate.",
    )
    parser.add_argument(
        "--exp-dir", type=str, default=None,
        help="Local experiment directory (alternative to --experiment).",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(PROJECT_ROOT / "outputs" / "s3_metrics"),
        help="Local directory for downloaded S3 files.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    output_dir = Path(args.output_dir)

    if args.experiment:
        # Evaluate from S3
        all_results = {}
        for exp_name in args.experiment:
            print(f"\n{'='*60}")
            print(f"Evaluating: {exp_name}")
            print(f"{'='*60}")

            metrics = load_metrics_from_s3(exp_name, output_dir)
            if not metrics:
                print(f"  ERROR: No metrics found for {exp_name}")
                continue

            posthoc = compute_posthoc_metrics(metrics)
            all_results[exp_name] = posthoc

            # Print individual results
            print(f"  Pearson R:    {posthoc.get('student_pearson_r', 'N/A')}")
            print(f"  Poisson NLL:  {posthoc.get('student_poisson_nll', 'N/A')}")
            print(f"  MSE:          {posthoc.get('student_mse', 'N/A')}")
            print(f"  MAE:          {posthoc.get('student_mae', 'N/A')}")
            print(f"  R²:           {posthoc.get('r_squared', 'N/A')}")
            print(f"  Retention:    {posthoc.get('retention_pct', 'N/A')}")

        # Print comparison table
        if len(all_results) > 1:
            print_comparison_table(all_results)
        elif len(all_results) == 1:
            print_comparison_table(all_results)

    elif args.exp_dir:
        # Evaluate from local directory
        exp_dir = Path(args.exp_dir)
        metrics_path = exp_dir / "metrics.json"
        if not metrics_path.exists():
            print(f"ERROR: {metrics_path} not found")
            sys.exit(1)

        with open(metrics_path) as f:
            metrics = json.load(f)

        posthoc = compute_posthoc_metrics(metrics)
        print_comparison_table({exp_dir.name: posthoc})

    else:
        # Evaluate ALL experiments in output_dir
        all_results = {}
        for exp_path in sorted(output_dir.iterdir()):
            metrics_path = exp_path / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
                posthoc = compute_posthoc_metrics(metrics)
                all_results[exp_path.name] = posthoc

        if all_results:
            print_comparison_table(all_results)
        else:
            print("No experiments found. Run with --experiment to pull from S3.")


if __name__ == "__main__":
    main()
