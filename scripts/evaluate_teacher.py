"""
Evaluate a trained teacher model on train/val/test splits.

Loads an experiment's checkpoint and config, rebuilds the data pipeline,
and runs inference on all three splits. This is a sanity check to verify
the model actually learned — train-set metrics should be better than
val/test.

Usage:
    python scripts/evaluate_teacher.py --exp-dir experiments/2026-02-16_steinmetz_ema_only
    python scripts/evaluate_teacher.py --exp-dir experiments/2026-02-16_steinmetz_ema_only --checkpoint final_model.pt
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path for `src.*` imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.spikeinterface_generator import generate_synthetic_recording
from src.data.modulated_generator import generate_modulated_spikes
from src.data.real_data_loader import load_nwb_spikes
from src.data.binning import bin_spike_trains
from src.data.spike_dataset import create_dataloaders
from src.models.teacher import TeacherLSTM
from src.train.trainer import Trainer
from src.utils.config import load_config
from src.utils.device import resolve_device, log_device_info

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained teacher model on all data splits.",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="Path to the experiment directory (contains config.yaml and checkpoints).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model.pt",
        help="Checkpoint filename to load (default: best_model.pt).",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save evaluation results as eval_metrics.json in the experiment dir.",
    )
    return parser.parse_args()


def _rebuild_data(data_config: dict) -> np.ndarray:
    """
    Regenerate or reload the spike-count matrix from the saved data config.

    This mirrors the data-loading logic in train_teacher.py so that we
    evaluate on exactly the same data the model was trained on.

    Args:
        data_config: The data configuration dict from the experiment.

    Returns:
        spike_counts array, shape (M, T_total).
    """
    source_type = data_config.get("source", {}).get("type", "spikeinterface")
    logger.info("Data source type: %s", source_type)

    if source_type == "modulated":
        sorting, _ = generate_modulated_spikes(data_config)
        recording = None
    elif source_type == "nwb":
        sorting, _ = load_nwb_spikes(data_config)
        recording = None
    else:
        recording, sorting, *_ = generate_synthetic_recording(data_config)

    bin_width_ms = data_config.get("bin_width_ms", 10.0)
    spike_counts, _ = bin_spike_trains(
        sorting, bin_width_ms=bin_width_ms, recording=recording,
    )
    return spike_counts


def main() -> None:
    """Main entrypoint: load config → rebuild data → load model → evaluate."""
    args = parse_args()
    exp_dir = Path(args.exp_dir)

    # ------------------------------------------------------------------
    # 1. Load the experiment config
    # ------------------------------------------------------------------
    config_path = exp_dir / "config.yaml"
    if not config_path.exists():
        logger.error("No config.yaml found in %s", exp_dir)
        sys.exit(1)

    combined_config = load_config(str(config_path))
    teacher_config = combined_config["teacher"]
    data_config = combined_config["data"]

    # Reproducibility: set the same seed used during training
    seed = teacher_config.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Set random seed: %d", seed)

    # ------------------------------------------------------------------
    # 2. Resolve device
    # ------------------------------------------------------------------
    compute_cfg = teacher_config.get("compute", {})
    device = resolve_device(compute_cfg.get("device", "auto"))
    log_device_info(device)

    # ------------------------------------------------------------------
    # 3. Rebuild data pipeline (identical to training)
    # ------------------------------------------------------------------
    logger.info("Rebuilding data pipeline...")
    spike_counts = _rebuild_data(data_config)
    m, t_total = spike_counts.shape
    logger.info("Spike counts: shape (%d, %d)", m, t_total)

    # Compute input_size including history features
    from src.data.history_features import compute_history_features
    hf_config = data_config.get("history_features", {})
    if hf_config.get("enabled", False):
        _, n_feat = compute_history_features(spike_counts, data_config)
        input_size = m + n_feat * m
        logger.info(
            "History features enabled: %d features × %d channels → "
            "input_size=%d", n_feat, m, input_size,
        )
    else:
        input_size = m

    # Ensure model config matches
    model_cfg = teacher_config.setdefault("model", {})
    model_cfg["input_size"] = input_size
    model_cfg["output_size"] = m

    # ------------------------------------------------------------------
    # 4. Create DataLoaders (same splits as training)
    # ------------------------------------------------------------------
    logger.info("Creating DataLoaders...")
    loaders = create_dataloaders(spike_counts, data_config)
    for split_name, loader in loaders.items():
        logger.info("  %s: %d batches", split_name, len(loader))

    # ------------------------------------------------------------------
    # 5. Load model from checkpoint
    # ------------------------------------------------------------------
    checkpoint_path = exp_dir / args.checkpoint
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        sys.exit(1)

    logger.info("Creating TeacherLSTM model...")
    model = TeacherLSTM.from_config(teacher_config, input_size=input_size)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d", n_params)

    logger.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(
        "Loaded checkpoint from epoch %d (best=%s)",
        checkpoint.get("epoch", -1),
        checkpoint.get("is_best", "?"),
    )

    # ------------------------------------------------------------------
    # 6. Evaluate on all splits
    # ------------------------------------------------------------------
    # Create a lightweight Trainer just for the evaluate() method
    trainer = Trainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        config=teacher_config,
        device=device,
    )

    results = {}
    for split_name, loader in loaders.items():
        logger.info("Evaluating on %s split...", split_name)
        metrics = trainer.evaluate(loader, prefix=split_name)
        results[split_name] = metrics
        for k, v in metrics.items():
            logger.info("  %s = %.6f", k, v)

    # ------------------------------------------------------------------
    # 7. Print comparison table
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("EVALUATION RESULTS")
    print("=" * 72)
    print(f"Experiment : {exp_dir}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Parameters : {n_params:,}")
    print("-" * 72)
    print(f"{'Metric':<25} {'Train':>12} {'Val':>12} {'Test':>12}")
    print("-" * 72)

    # Metric suffixes to display
    metric_suffixes = ["loss", "poisson_nll", "pearson_r", "mae", "mse"]
    for suffix in metric_suffixes:
        train_val = results["train"].get(f"train_{suffix}", float("nan"))
        val_val = results["val"].get(f"val_{suffix}", float("nan"))
        test_val = results["test"].get(f"test_{suffix}", float("nan"))
        print(f"  {suffix:<23} {train_val:>12.6f} {val_val:>12.6f} {test_val:>12.6f}")

    print("=" * 72)

    # Sanity check: is the model learning?
    train_nll = results["train"].get("train_poisson_nll", float("inf"))
    val_nll = results["val"].get("val_poisson_nll", float("inf"))
    if train_nll < val_nll:
        print("[OK] Train NLL < Val NLL -> model is learning (expected)")
    else:
        print("[!!] Train NLL >= Val NLL -> model may NOT be learning!")
    print()

    # ------------------------------------------------------------------
    # 8. Optionally save results
    # ------------------------------------------------------------------
    if args.save:
        out_path = exp_dir / "eval_metrics.json"
        # Flatten for JSON
        flat = {}
        for split_name, metrics in results.items():
            flat.update(metrics)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(flat, f, indent=2)
        logger.info("Saved evaluation metrics to %s", out_path)


if __name__ == "__main__":
    main()
