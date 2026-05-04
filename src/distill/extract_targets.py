"""
Distillation target extraction.

Runs the trained teacher ANN over the full dataset and caches its
rate predictions (soft labels) alongside the ground-truth spike counts.
These cached targets are used by the student SNN during distillation
training, avoiding repeated teacher forward passes.

Output format per split (train/val/test):
    {
        "inputs":        Tensor (N, T, M)  — history windows
        "teacher_rates": Tensor (N, M)     — teacher λ^ANN predictions
        "targets":       Tensor (N, M)     — ground-truth y(t+1)
    }

Usage:
    from src.distill.extract_targets import extract_teacher_targets

    targets = extract_teacher_targets(model, loaders, device)
    save_distillation_targets(targets, output_dir)
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@torch.no_grad()
def extract_teacher_targets(
    model: torch.nn.Module,
    loaders: Dict[str, DataLoader],
    device: torch.device,
    splits: Optional[list] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Run the teacher model over data splits and collect soft labels.

    The teacher is set to eval mode. All outputs are collected on CPU
    to avoid GPU memory pressure on large datasets.

    Args:
        model: Trained teacher model (must output non-negative rates).
        loaders: Dict of DataLoaders, keyed by split name
                 (e.g., "train", "val", "test").
        device: Device to run inference on.
        splits: Optional list of split names to extract. If None,
                extracts all splits found in loaders.

    Returns:
        Dict mapping split name → dict with keys:
            "inputs"        : Tensor (N, T, M) — history windows
            "teacher_rates" : Tensor (N, M) — predicted rates
            "targets"       : Tensor (N, M) — ground-truth counts
    """
    model.eval()
    model.to(device)

    if splits is None:
        splits = list(loaders.keys())

    results = {}

    for split_name in splits:
        if split_name not in loaders:
            logger.warning("Split '%s' not found in loaders, skipping.", split_name)
            continue

        loader = loaders[split_name]
        all_inputs = []
        all_rates = []
        all_targets = []

        logger.info(
            "Extracting teacher targets for '%s' split (%d batches)...",
            split_name, len(loader),
        )

        for x, y in loader:
            # Forward pass on device
            x_dev = x.to(device)
            rates = model(x_dev)

            # Collect on CPU to save GPU memory
            all_inputs.append(x.cpu())
            all_rates.append(rates.cpu())
            all_targets.append(y.cpu())

        # Concatenate along batch dimension
        results[split_name] = {
            "inputs": torch.cat(all_inputs, dim=0),
            "teacher_rates": torch.cat(all_rates, dim=0),
            "targets": torch.cat(all_targets, dim=0),
        }

        n_samples = results[split_name]["inputs"].shape[0]
        n_channels = results[split_name]["teacher_rates"].shape[1]
        logger.info(
            "  %s: %d samples, %d channels, rates range [%.4f, %.4f]",
            split_name, n_samples, n_channels,
            results[split_name]["teacher_rates"].min().item(),
            results[split_name]["teacher_rates"].max().item(),
        )

    return results


def save_distillation_targets(
    targets: Dict[str, Dict[str, torch.Tensor]],
    output_dir: Union[str, Path],
) -> Path:
    """
    Save extracted distillation targets to disk.

    Creates one .pt file per split containing the inputs, teacher
    rates, and ground-truth targets.

    Args:
        targets: Dict from extract_teacher_targets().
        output_dir: Directory to save target files.

    Returns:
        Path to the output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, data in targets.items():
        save_path = output_dir / f"distill_targets_{split_name}.pt"
        torch.save(data, save_path)

        n_samples = data["inputs"].shape[0]
        size_mb = save_path.stat().st_size / (1024 ** 2)
        logger.info(
            "Saved %s targets: %d samples → %s (%.1f MB)",
            split_name, n_samples, save_path, size_mb,
        )

    # Save a metadata file for reproducibility
    metadata = {
        "splits": list(targets.keys()),
        "shapes": {
            name: {k: list(v.shape) for k, v in data.items()}
            for name, data in targets.items()
        },
    }
    meta_path = output_dir / "distill_targets_metadata.pt"
    torch.save(metadata, meta_path)
    logger.info("Saved metadata to %s", meta_path)

    return output_dir


def load_distillation_targets(
    input_dir: Union[str, Path],
    split: str = "train",
) -> Dict[str, torch.Tensor]:
    """
    Load previously saved distillation targets for a given split.

    Args:
        input_dir: Directory containing saved target files.
        split: Which split to load ("train", "val", or "test").

    Returns:
        Dict with keys "inputs", "teacher_rates", "targets".

    Raises:
        FileNotFoundError: If the target file does not exist.
    """
    input_dir = Path(input_dir)
    target_path = input_dir / f"distill_targets_{split}.pt"

    if not target_path.exists():
        raise FileNotFoundError(
            f"Distillation targets not found: {target_path}. "
            f"Run extract_teacher_targets first."
        )

    data = torch.load(target_path, weights_only=True)
    logger.info(
        "Loaded %s targets: inputs=%s, teacher_rates=%s, targets=%s",
        split,
        list(data["inputs"].shape),
        list(data["teacher_rates"].shape),
        list(data["targets"].shape),
    )
    return data


def validate_distillation_targets(
    targets: Dict[str, Dict[str, torch.Tensor]],
) -> Dict[str, Any]:
    """
    Validate extracted distillation targets for consistency.

    Checks:
    - All splits have matching channel dimensions
    - Teacher rates are non-negative
    - Shapes are consistent (N, T, M) for inputs, (N, M) for rates/targets
    - No NaN or Inf values

    Args:
        targets: Dict from extract_teacher_targets().

    Returns:
        Dict with validation statistics per split.

    Raises:
        ValueError: If any validation check fails.
    """
    stats = {}

    # Collect channel dims for cross-split consistency
    channel_dims = {}

    for split_name, data in targets.items():
        inputs = data["inputs"]
        rates = data["teacher_rates"]
        ground_truth = data["targets"]

        n_samples = inputs.shape[0]
        t_hist = inputs.shape[1]
        m_channels = inputs.shape[2]

        # Shape consistency
        if rates.shape != (n_samples, m_channels):
            raise ValueError(
                f"{split_name}: teacher_rates shape {rates.shape} doesn't "
                f"match expected ({n_samples}, {m_channels})"
            )
        if ground_truth.shape != (n_samples, m_channels):
            raise ValueError(
                f"{split_name}: targets shape {ground_truth.shape} doesn't "
                f"match expected ({n_samples}, {m_channels})"
            )

        # Non-negative rates (teacher uses softplus, so should always be > 0)
        if (rates < 0).any():
            raise ValueError(
                f"{split_name}: teacher_rates contain negative values "
                f"(min={rates.min().item():.6f})"
            )

        # No NaN/Inf
        for name, tensor in [("inputs", inputs), ("rates", rates), ("targets", ground_truth)]:
            if torch.isnan(tensor).any():
                raise ValueError(f"{split_name}/{name}: contains NaN values")
            if torch.isinf(tensor).any():
                raise ValueError(f"{split_name}/{name}: contains Inf values")

        # Record channel dim for cross-split check
        channel_dims[split_name] = m_channels

        stats[split_name] = {
            "n_samples": n_samples,
            "history_bins": t_hist,
            "n_channels": m_channels,
            "rate_min": rates.min().item(),
            "rate_max": rates.max().item(),
            "rate_mean": rates.mean().item(),
            "target_mean": ground_truth.mean().item(),
        }

        logger.info(
            "Validated %s: %d samples, T=%d, M=%d, "
            "rates=[%.4f, %.4f], target_mean=%.4f",
            split_name, n_samples, t_hist, m_channels,
            rates.min().item(), rates.max().item(),
            ground_truth.mean().item(),
        )

    # Cross-split channel consistency
    unique_channels = set(channel_dims.values())
    if len(unique_channels) > 1:
        raise ValueError(
            f"Channel dimension mismatch across splits: {channel_dims}"
        )

    return stats
