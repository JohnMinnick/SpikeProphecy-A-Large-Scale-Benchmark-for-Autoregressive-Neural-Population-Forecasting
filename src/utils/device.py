"""
Device resolution and GPU utility functions.

Provides a single entry point for resolving compute devices from config
values. All device logic in the project should go through this module
rather than calling torch.device() directly.

Usage:
    from src.utils.device import resolve_device, log_device_info

    device = resolve_device("auto")   # Resolves to best available
    log_device_info(device)           # Logs GPU name, VRAM, etc.
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def resolve_device(device_config: str = "auto") -> torch.device:
    """
    Resolve a config device string to a torch.device.

    Args:
        device_config: One of:
            - "auto"   : Use CUDA if available, else CPU.
            - "cpu"    : Force CPU.
            - "cuda"   : Use default CUDA device.
            - "cuda:N" : Use specific CUDA device N.

    Returns:
        torch.device: Resolved PyTorch device.

    Raises:
        RuntimeError: If a CUDA device is requested but not available.
    """
    config = device_config.strip().lower()

    if config == "auto":
        # Auto-detect: prefer CUDA if available
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    elif config == "cpu":
        device = torch.device("cpu")
    elif config.startswith("cuda"):
        # Handles "cuda" and "cuda:N"
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device '{device_config}' but CUDA is not available. "
                "Install a CUDA-enabled PyTorch or use device: 'cpu'."
            )
        device = torch.device(config)
    else:
        raise ValueError(
            f"Unknown device config: '{device_config}'. "
            "Expected one of: 'auto', 'cpu', 'cuda', 'cuda:N'."
        )

    logger.info("Resolved device: %s (from config: '%s')", device, device_config)
    return device


def log_device_info(device: torch.device) -> None:
    """
    Log detailed information about the resolved compute device.

    For CUDA devices, logs: device name, total VRAM, CUDA version,
    and cuDNN version. For CPU, logs basic info.

    Args:
        device: The resolved torch.device to log info about.
    """
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        gpu_name = torch.cuda.get_device_name(idx)
        vram_bytes = torch.cuda.get_device_properties(idx).total_memory
        vram_gb = vram_bytes / (1024 ** 3)
        cuda_version = torch.version.cuda or "unknown"
        cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"

        logger.info("GPU %d: %s (%.1f GB VRAM)", idx, gpu_name, vram_gb)
        logger.info("CUDA version: %s, cuDNN version: %s", cuda_version, cudnn_version)
    else:
        logger.info("Running on CPU")


def get_num_gpus() -> int:
    """
    Return the number of available CUDA GPUs.

    Returns:
        int: Number of CUDA devices visible to PyTorch.
    """
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    logger.debug("Available GPUs: %d", count)
    return count


def get_default_num_workers(device: torch.device, override: Optional[int] = None) -> int:
    """
    Suggest a reasonable num_workers for DataLoader based on device.

    If an override is provided (from config), it is returned directly.
    Otherwise, returns 4 for CUDA devices and 0 for CPU (to avoid
    multiprocessing overhead on small machines).

    Args:
        device: The resolved compute device.
        override: Optional explicit value from config.

    Returns:
        int: Number of DataLoader workers to use.
    """
    if override is not None:
        return override

    # Default heuristic: 4 workers for GPU, 0 for CPU
    return 4 if device.type == "cuda" else 0
