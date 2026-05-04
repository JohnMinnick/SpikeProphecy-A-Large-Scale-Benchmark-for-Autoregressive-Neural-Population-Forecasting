"""
DEPRECATED — FGL (Future-Guided Learning) Dataset wrapper.

⚠️ This module is LEGACY. FGL was superseded by standard distillation.
Kept for test compatibility. See ADR-0014 for history.

Wraps an existing SpikeCountDataset to produce 4-tuples for FGL training:
  - x_student: causal history window [t-T .. t]
  - x_teacher: future-privileged window [t-T+K .. t+K]
  - y_target: ground truth at t+K (the target both models predict)
  - (optional) session metadata

The teacher sees K bins into the future relative to the student,
giving it privileged information that it distills into the student
via Poisson KL divergence.

Architecture:
    Student: See [t-T .. t     ] → predict y(t+1)    # causal
    Teacher: See [t-T+K .. t+K ] → predict y(t+K+1)  # privileged

    But both predict the SAME target: y(t+K+1).
    The student must learn to predict further ahead without seeing the future.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class FGLDataset(Dataset):
    """
    Dataset wrapper that adds temporal offset for Future-Guided Learning.

    Given a spike-count time series (T_total, M) in time-first layout,
    produces student/teacher window pairs offset by K bins.

    Args:
        data: Spike-count tensor, shape (T_total, M), time-first.
        history_bins: Number of history bins (T) for input windows.
        K: Temporal offset — teacher sees K bins into the future.
        num_channels: Number of real spike-count channels (M).
            Used to separate targets from any appended features.
    """

    def __init__(
        self,
        data: torch.Tensor,
        history_bins: int,
        K: int = 5,
        num_channels: Optional[int] = None,
    ):
        super().__init__()

        self.data = data
        self.history_bins = history_bins
        self.K = K
        self.total_bins = data.shape[0]
        self.num_channels = num_channels or data.shape[1]

        # Valid samples: student window [idx .. idx+T] and teacher window
        # [idx+K .. idx+K+T] must fit, plus target at idx+K+T.
        # Student window: idx to idx + T - 1 (T bins)
        # Teacher window: idx + K to idx + K + T - 1 (T bins)
        # Target: idx + K + T (1 bin)
        # Max idx: total_bins - K - T - 1
        self.num_samples = self.total_bins - self.K - self.history_bins

        if self.num_samples <= 0:
            raise ValueError(
                f"Not enough data: T_total={self.total_bins}, T={history_bins}, "
                f"K={K} → {self.num_samples} samples (need > 0). "
                f"Reduce K or T, or use longer recordings."
            )

        logger.info(
            "FGLDataset: T=%d, K=%d, M=%d, %d samples "
            "(reduced from %d by K=%d offset)",
            history_bins, K, self.num_channels, self.num_samples,
            self.total_bins - history_bins, K,
        )

    def __len__(self) -> int:
        """Return number of valid FGL samples."""
        return self.num_samples

    def __getitem__(
        self, idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get (student_input, teacher_input, target) triplet at index idx.

        Args:
            idx: Sample index in [0, num_samples).

        Returns:
            Tuple of:
                - x_student: (T, M) — causal history window
                - x_teacher: (T, M) — future-privileged window (shifted by K)
                - y_target: (M_channels,) — ground truth at t+K+T
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(
                f"Index {idx} out of range [0, {self.num_samples})"
            )

        T = self.history_bins

        # Student: causal window [idx .. idx + T)
        x_student = self.data[idx : idx + T, :]

        # Teacher: future-privileged window [idx + K .. idx + K + T)
        x_teacher = self.data[idx + self.K : idx + self.K + T, :]

        # Target: ground truth at the SAME bin = idx + K + T
        # (both student and teacher predict this same target)
        # Use ALL channels (M_max when padded) so ConcatDataset batches
        # from sessions with different neuron counts produce uniform
        # tensor sizes.  The trainer's loss masking handles zero-padded
        # channels downstream.
        y_target = self.data[idx + self.K + T, :]

        return x_student, x_teacher, y_target


def create_fgl_dataloaders(
    spike_counts: np.ndarray,
    config: dict,
    K: int = 5,
    batch_size: int = 512,
    num_workers: int = 0,
) -> dict:
    """
    Create FGL train/val/test DataLoaders from a spike-count matrix.

    Performs temporal splitting first, then wraps each split with
    FGLDataset for the temporal offset.

    Args:
        spike_counts: Shape (M, T_total).
        config: Data config with splits and history_bins.
        K: FGL temporal offset.
        batch_size: Batch size for DataLoaders.
        num_workers: Number of DataLoader workers.

    Returns:
        Dict with 'train', 'val', 'test' DataLoaders.
    """
    from src.data.spike_dataset import temporal_split

    # Split temporally
    splits = config.get("splits", {})
    train_counts, val_counts, test_counts = temporal_split(
        spike_counts,
        train_ratio=splits.get("train", 0.7),
        val_ratio=splits.get("val", 0.15),
        test_ratio=splits.get("test", 0.15),
    )

    history_bins = config.get("history_bins", 10)
    dtype = torch.float32

    loaders = {}
    for name, counts in [
        ("train", train_counts),
        ("val", val_counts),
        ("test", test_counts),
    ]:
        # Convert to time-first tensor: (M, T) -> (T, M)
        data_tensor = torch.tensor(counts.T, dtype=dtype)
        num_channels = counts.shape[0]

        dataset = FGLDataset(
            data=data_tensor,
            history_bins=history_bins,
            K=K,
            num_channels=num_channels,
        )
        loaders[name] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )

    return loaders
