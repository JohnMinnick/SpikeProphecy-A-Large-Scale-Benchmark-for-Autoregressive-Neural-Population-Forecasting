"""
SHD (Spiking Heidelberg Digits) dataset loader.

Downloads and preprocesses the SHD dataset for classification tasks.
SHD contains 700-channel spike-train recordings of spoken digits (0-9)
in English and German (20 classes total).

The raw data contains spike events (neuron_id, time). This loader bins
the events into fixed-width time bins to produce dense tensors suitable
for Mamba and SNN models.

Dataset source: https://zenkelab.org/resources/spiking-heidelberg-digits-shd/
Reference: Cramer et al., "The Heidelberg Spiking Datasets" (2020)
"""

import logging
import os
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# SHD dataset URLs (from zenkelab)
SHD_URLS = {
    "train": "https://zenkelab.org/datasets/shd_train.h5.gz",
    "test": "https://zenkelab.org/datasets/shd_test.h5.gz",
}

# SHD constants
SHD_NUM_CHANNELS = 700  # Number of input neurons (cochlea channels)
SHD_NUM_CLASSES = 20    # Digits 0-9 in English + German
SHD_MAX_TIME = 1.4      # Maximum event time in seconds


class SHDDataset(Dataset):
    """
    Spiking Heidelberg Digits dataset.

    Bins raw spike events into fixed-width time bins to produce
    dense spike count tensors of shape (T, num_channels).

    Args:
        data_dir: Directory to download/cache the dataset.
        split: 'train' or 'test'.
        bin_size_ms: Time bin width in milliseconds (default 10ms).
        max_time_ms: Maximum sequence length in ms (default 1000ms = 1s).
            Sequences shorter than this are zero-padded.
            Sequences longer are truncated.
        transform: Optional transform applied to each sample.
    """

    def __init__(
        self,
        data_dir: str = "data/shd",
        split: str = "train",
        bin_size_ms: float = 10.0,
        max_time_ms: float = 1000.0,
        transform=None,
    ):
        assert split in ("train", "test"), f"split must be 'train' or 'test', got {split}"
        self.data_dir = Path(data_dir)
        self.split = split
        self.bin_size_ms = bin_size_ms
        self.max_time_ms = max_time_ms
        self.transform = transform

        # Derived parameters
        self.num_bins = int(max_time_ms / bin_size_ms)  # T dimension
        self.num_channels = SHD_NUM_CHANNELS
        self.num_classes = SHD_NUM_CLASSES

        # Download if needed and load
        self._ensure_downloaded()
        self.spike_data, self.labels = self._load_and_bin()

        logger.info(
            "SHD %s: %d samples, T=%d bins (%.0fms @ %.0fms), "
            "%d channels, %d classes",
            split, len(self.labels), self.num_bins,
            max_time_ms, bin_size_ms, self.num_channels, self.num_classes,
        )

    def _ensure_downloaded(self):
        """Download SHD HDF5 files if not present."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        h5_path = self.data_dir / f"shd_{self.split}.h5"

        if h5_path.exists():
            return

        gz_path = self.data_dir / f"shd_{self.split}.h5.gz"
        url = SHD_URLS[self.split]

        if not gz_path.exists():
            logger.info("Downloading SHD %s from %s...", self.split, url)
            urllib.request.urlretrieve(url, str(gz_path))
            logger.info("Downloaded: %s", gz_path)

        # Decompress gzip
        import gzip
        logger.info("Decompressing %s...", gz_path.name)
        with gzip.open(str(gz_path), 'rb') as f_in:
            with open(str(h5_path), 'wb') as f_out:
                # Read in chunks to handle large files
                while True:
                    chunk = f_in.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    f_out.write(chunk)
        logger.info("Decompressed to: %s", h5_path)

    def _load_and_bin(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load HDF5 spike data and bin into fixed-width time bins.

        SHD HDF5 structure (variable-length datasets):
        - spikes/times: VLEN dataset — each entry is a variable-length
          array of spike times for one sample
        - spikes/units: VLEN dataset — each entry is a variable-length
          array of neuron IDs for one sample
        - labels: 1D array of integer class labels

        Returns:
            spike_data: (N, T, C) float32 tensor of binned spike counts
            labels: (N,) long tensor of class labels
        """
        h5_path = self.data_dir / f"shd_{self.split}.h5"
        bin_size_s = self.bin_size_ms / 1000.0

        with h5py.File(str(h5_path), 'r') as f:
            # Read VLEN datasets: returns numpy object arrays of ragged arrays
            times_data = f['spikes']['times'][()]
            units_data = f['spikes']['units'][()]
            labels_arr = np.array(f['labels'], dtype=np.int64)

        num_samples = len(labels_arr)
        # Pre-allocate output tensor
        spike_data = torch.zeros(
            num_samples, self.num_bins, self.num_channels,
            dtype=torch.float32,
        )

        for i in range(num_samples):
            times = np.asarray(times_data[i], dtype=np.float64)
            units = np.asarray(units_data[i], dtype=np.int64)

            # Compute bin indices from spike times
            bin_indices = np.floor(times / bin_size_s).astype(np.int64)

            # Filter out-of-range spikes
            valid = (bin_indices >= 0) & (bin_indices < self.num_bins) & \
                    (units >= 0) & (units < self.num_channels)
            bin_indices = bin_indices[valid]
            units = units[valid]

            # Accumulate spike counts
            if len(bin_indices) > 0:
                # Use np.add.at for unbuffered in-place addition
                data_np = spike_data[i].numpy()
                np.add.at(data_np, (bin_indices, units), 1.0)
                spike_data[i] = torch.from_numpy(data_np)

        labels = torch.from_numpy(labels_arr)
        return spike_data, labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.

        Returns:
            x: (T, C) binned spike counts
            y: integer class label
        """
        x = self.spike_data[idx]
        y = self.labels[idx].item()

        if self.transform is not None:
            x = self.transform(x)

        return x, y


def create_shd_loaders(
    data_dir: str = "data/shd",
    bin_size_ms: float = 10.0,
    max_time_ms: float = 1000.0,
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create SHD train and test data loaders.

    Args:
        data_dir: Directory for dataset download/cache.
        bin_size_ms: Time bin width in milliseconds.
        max_time_ms: Maximum sequence length.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        pin_memory: Pin memory for GPU transfer.

    Returns:
        (train_loader, test_loader) tuple.
    """
    train_ds = SHDDataset(
        data_dir=data_dir, split="train",
        bin_size_ms=bin_size_ms, max_time_ms=max_time_ms,
    )
    test_ds = SHDDataset(
        data_dir=data_dir, split="test",
        bin_size_ms=bin_size_ms, max_time_ms=max_time_ms,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, test_loader
