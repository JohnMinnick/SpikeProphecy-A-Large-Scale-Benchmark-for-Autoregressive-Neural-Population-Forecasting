"""
Reproducibility seeding utilities.

Provides a single function to seed all random number generators used in
the project (Python, NumPy, PyTorch CPU, and PyTorch CUDA) and log the
seed and determinism settings.

Usage:
    from src.utils.seed import seed_everything

    seed_everything(42)  # Seeds all RNGs, logs settings
"""

import logging
import os
import random
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42, deterministic: bool = False) -> int:
    """
    Seed all random number generators for reproducibility.

    Sets seeds for: Python's random module, NumPy, PyTorch CPU, and
    PyTorch CUDA (if available). Optionally enables PyTorch deterministic
    mode for fully reproducible results (at a potential performance cost).

    Args:
        seed: Integer seed value. Must be non-negative.
        deterministic: If True, enables PyTorch deterministic algorithms
            and disables cuDNN benchmarking. This can significantly slow
            down training but guarantees bitwise reproducibility.

    Returns:
        int: The seed that was set (useful if you want to log it).

    Raises:
        ValueError: If seed is negative.
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")

    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Deterministic mode (optional, affects performance)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        # Set CUBLAS workspace config for deterministic behavior
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        # Allow cuDNN auto-tuner for best performance
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True

    # Log the seed and determinism settings
    logger.info("Seed set to %d", seed)
    logger.info(
        "Deterministic mode: %s | cuDNN benchmark: %s",
        deterministic,
        not deterministic if torch.backends.cudnn.is_available() else "N/A",
    )

    return seed


def get_seed_from_config(config: dict) -> int:
    """
    Extract seed from a config dictionary.

    Looks for a top-level 'seed' key. Returns 42 as a default if not
    found, and logs a warning.

    Args:
        config: Configuration dictionary (parsed YAML).

    Returns:
        int: The seed value.
    """
    seed = config.get("seed", None)
    if seed is None:
        seed = 42
        logger.warning("No 'seed' found in config, defaulting to %d", seed)
    return int(seed)
