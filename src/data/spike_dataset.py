"""
PyTorch Dataset for spike-count forecasting.

Converts a binned spike-count matrix (M × T_total) into a PyTorch Dataset
that yields (X_t, y_{t+1}) tuples for training the teacher ANN.

  - X_t: (T, M) or (T, M + N*M) tensor — history window, optionally
         augmented with spike-history features (ISI, EMA, refractory).
  - y_{t+1}: (M,) tensor — next-step spike counts to predict

Also provides functions for temporal train/val/test splitting and
DataLoader creation with GPU-aware settings.

Usage:
    from src.data.spike_dataset import SpikeCountDataset, create_dataloaders

    dataset = SpikeCountDataset(spike_counts, history_bins=50)
    loaders = create_dataloaders(spike_counts, config)
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class SpikeCountDataset(Dataset):
    """
    PyTorch Dataset for spike-count time-series forecasting.

    Given a spike-count matrix of shape (M, T_total), this dataset creates
    sliding-window samples:
        - Input X_t: history of T bins, shape (T, M)  [time-first for RNNs]
        - Target y_{t+1}: next-step counts, shape (M,)

    The total number of samples is T_total - T (the first T bins are consumed
    by the first window, and each subsequent sample slides by 1 bin).

    Attributes:
        spike_counts: Raw count matrix (M, T_total).
        history_bins: Number of history bins (T) per sample.
        num_samples: Total number of (input, target) pairs.
    """

    def __init__(
        self,
        spike_counts: np.ndarray,
        history_bins: int = 50,
        forecast_horizon: int = 1,
        dtype: torch.dtype = torch.float32,
        history_feature_matrix: Optional[np.ndarray] = None,
    ):
        """
        Args:
            spike_counts: Spike-count matrix, shape (M, T_total), dtype int32.
            history_bins: Number of history bins (T) for the input window.
            forecast_horizon: Number of future bins to predict (H). Currently
                only H=1 is supported.
            dtype: Torch dtype for output tensors.
            history_feature_matrix: Optional feature matrix, shape
                (N*M, T_total). If provided, appended as extra channels
                to inputs. Targets remain spike counts only (M,).
        """
        super().__init__()

        if spike_counts.ndim != 2:
            raise ValueError(
                f"spike_counts must be 2D (M, T_total), got shape {spike_counts.shape}"
            )

        self.num_channels, self.total_bins = spike_counts.shape

        if history_bins < 1:
            raise ValueError(f"history_bins must be >= 1, got {history_bins}")
        if history_bins >= self.total_bins:
            raise ValueError(
                f"history_bins ({history_bins}) must be < total_bins ({self.total_bins})"
            )
        if forecast_horizon != 1:
            raise NotImplementedError(
                "Only forecast_horizon=1 is supported in Iteration 1. "
                f"Got forecast_horizon={forecast_horizon}"
            )

        self.history_bins = history_bins
        self.forecast_horizon = forecast_horizon
        self.dtype = dtype

        # Validate and incorporate history features if provided
        if history_feature_matrix is not None:
            if history_feature_matrix.ndim != 2:
                raise ValueError(
                    f"history_feature_matrix must be 2D, got shape "
                    f"{history_feature_matrix.shape}"
                )
            if history_feature_matrix.shape[1] != self.total_bins:
                raise ValueError(
                    f"history_feature_matrix T_total ({history_feature_matrix.shape[1]}) "
                    f"must match spike_counts T_total ({self.total_bins})"
                )
            self.num_history_features = history_feature_matrix.shape[0]
            # Stack: counts (M, T) + features (N*M, T) → (M + N*M, T)
            combined = np.concatenate(
                [spike_counts, history_feature_matrix], axis=0
            )
        else:
            self.num_history_features = 0
            combined = spike_counts

        # Total input dimension: M + N*M
        self.input_size = combined.shape[0]

        # Convert to tensor once (transpose to time-first for slicing)
        # Shape: (T_total, M + N*M) — time steps along axis 0
        self._data = torch.tensor(
            combined.T, dtype=dtype,
        )

        # Number of valid (input, target) pairs
        self.num_samples = self.total_bins - self.history_bins

        logger.info(
            "SpikeCountDataset: %d channels + %d history features = %d input dims, "
            "%d total bins, T=%d history, %d samples",
            self.num_channels, self.num_history_features,
            self.input_size, self.total_bins,
            self.history_bins, self.num_samples,
        )

    def __len__(self) -> int:
        """Return number of samples."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the (input, target) pair at index idx.

        Args:
            idx: Sample index in [0, num_samples).

        Returns:
            Tuple of:
                - x: (T, M + N*M) float tensor — history window
                      (includes history features if enabled)
                - y: (M,) float tensor — next-step target
                      (always original spike counts only)
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(
                f"Index {idx} out of range [0, {self.num_samples})"
            )

        # Input window: bins [idx, idx + T), all channels + features
        x = self._data[idx : idx + self.history_bins, :]
        # Target: bin at idx + T, spike counts only (first M channels)
        y = self._data[idx + self.history_bins, :self.num_channels]

        return x, y


def temporal_split(
    spike_counts: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a spike-count matrix into train/val/test by time.

    Uses temporal (non-shuffled) splitting to prevent data leakage from
    future bins into training data.

    Args:
        spike_counts: Shape (M, T_total).
        train_ratio: Fraction of bins for training.
        val_ratio: Fraction of bins for validation.
        test_ratio: Fraction of bins for testing.

    Returns:
        Tuple of (train_counts, val_counts, test_counts), each (M, T_split).
    """
    # Validate ratios sum to ~1.0
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.3f} "
            f"({train_ratio} + {val_ratio} + {test_ratio})"
        )

    t_total = spike_counts.shape[1]
    train_end = int(t_total * train_ratio)
    val_end = train_end + int(t_total * val_ratio)

    train = spike_counts[:, :train_end]
    val = spike_counts[:, train_end:val_end]
    test = spike_counts[:, val_end:]

    logger.info(
        "Temporal split: train=%d bins (%.0f%%), val=%d bins (%.0f%%), "
        "test=%d bins (%.0f%%)",
        train.shape[1], 100 * train.shape[1] / t_total,
        val.shape[1], 100 * val.shape[1] / t_total,
        test.shape[1], 100 * test.shape[1] / t_total,
    )

    return train, val, test


def create_dataloaders(
    spike_counts: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders from a spike-count matrix.

    Reads history_bins, forecast_horizon, splits, and compute settings
    from the config dictionary. If history_features are enabled in config,
    computes features on the full matrix BEFORE splitting to ensure warm
    state at split boundaries.

    Args:
        spike_counts: Shape (M, T_total).
        config: Data configuration dictionary.

    Returns:
        Dict with keys 'train', 'val', 'test', each a DataLoader.
    """
    from src.data.history_features import compute_history_features

    # Extract config values
    history_bins = config.get("history_bins", 50)
    forecast_horizon = config.get("forecast_horizon", 1)
    splits = config.get("splits", {"train": 0.7, "val": 0.15, "test": 0.15})
    compute = config.get("compute", {})
    batch_size = config.get("batch_size", 64)
    num_workers = compute.get("num_workers", 0)
    pin_memory = compute.get("pin_memory", False)

    # Compute history features on the full matrix before splitting
    # so EMA/ISI state is warm at val/test boundaries
    feature_matrix, n_feat = compute_history_features(spike_counts, config)
    has_features = n_feat > 0

    # Split both counts and features identically by time
    train_counts, val_counts, test_counts = temporal_split(
        spike_counts,
        train_ratio=splits["train"],
        val_ratio=splits["val"],
        test_ratio=splits["test"],
    )

    # Split feature matrix along the same time boundaries
    if has_features:
        t_total = spike_counts.shape[1]
        train_end = int(t_total * splits["train"])
        val_end = train_end + int(t_total * splits["val"])

        train_feat = feature_matrix[:, :train_end]
        val_feat = feature_matrix[:, train_end:val_end]
        test_feat = feature_matrix[:, val_end:]
    else:
        train_feat = val_feat = test_feat = None

    # Create datasets with optional feature matrices
    train_ds = SpikeCountDataset(
        train_counts, history_bins, forecast_horizon,
        history_feature_matrix=train_feat,
    )
    val_ds = SpikeCountDataset(
        val_counts, history_bins, forecast_horizon,
        history_feature_matrix=val_feat,
    )
    test_ds = SpikeCountDataset(
        test_counts, history_bins, forecast_horizon,
        history_feature_matrix=test_feat,
    )

    # Create DataLoaders
    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,       # Shuffle training data only
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,     # Drop incomplete last batch for training
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    logger.info(
        "DataLoaders created: train=%d samples (%d batches), "
        "val=%d samples, test=%d samples, batch_size=%d, "
        "history_features=%d per channel",
        len(train_ds), len(loaders["train"]),
        len(val_ds), len(test_ds), batch_size, n_feat,
    )

    return loaders
