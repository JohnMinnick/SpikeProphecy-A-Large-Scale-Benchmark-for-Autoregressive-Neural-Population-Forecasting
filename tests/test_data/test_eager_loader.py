"""
Tests for EagerSessionCyclingLoader and create_dataloaders factory.

Validates that the eager loader:
- Pre-loads all sessions at init
- Produces the same batch shapes and masks as the lazy loader
- Supports session shuffling
- Integrates with the unified create_dataloaders() factory
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.multi_session_loader import (
    EagerSessionCyclingLoader,
    SessionCyclingLoader,
    create_dataloaders,
    create_eager_dataloaders,
    create_lazy_dataloaders,
    pad_to_channels,
    build_channel_mask,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cached_sessions(tmp_path):
    """
    Create a minimal multi-session cache on disk (3 sessions).

    Session 0: 4 channels, 200 bins
    Session 1: 6 channels, 150 bins
    Session 2: 3 channels, 180 bins

    Returns (cache_dir, metadata, config) for loader construction.
    """
    np.random.seed(42)

    sessions = [
        {"num_units": 4, "num_bins": 200},
        {"num_units": 6, "num_bins": 150},
        {"num_units": 3, "num_bins": 180},
    ]

    m_max = max(s["num_units"] for s in sessions)
    history_bins = 10

    session_info = []
    for i, s in enumerate(sessions):
        m_i, t_i = s["num_units"], s["num_bins"]

        # Generate random spike counts (uint8)
        counts = np.random.randint(0, 5, size=(m_i, t_i), dtype=np.uint8)
        npy_path = tmp_path / f"session_{i:03d}.npy"
        np.save(npy_path, counts)

        # Compute split boundaries (70/15/15)
        train_end = int(t_i * 0.7)
        val_end = train_end + int(t_i * 0.15)

        session_info.append({
            "index": i,
            "file": f"session_{i}.nwb",
            "npy_file": str(npy_path),
            "num_units": m_i,
            "num_bins": t_i,
            "split_boundaries": {
                "train_end": train_end,
                "val_end": val_end,
            },
        })

    metadata = {
        "num_sessions": len(sessions),
        "m_max": m_max,
        "history_bins": history_bins,
        "bin_width_ms": 10,
        "sessions": session_info,
    }

    # Write metadata
    with open(tmp_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    config = {
        "batch_size": 16,
        "history_bins": history_bins,
        "compute": {
            "num_workers": 0,
            "pin_memory": False,
        },
    }

    return tmp_path, metadata, config


# ---------------------------------------------------------------------------
# EagerSessionCyclingLoader tests
# ---------------------------------------------------------------------------

class TestEagerSessionCyclingLoader:
    """Tests for EagerSessionCyclingLoader."""

    def test_pre_loads_all_sessions(self, cached_sessions):
        """Eager loader should pre-build datasets for all valid sessions."""
        cache_dir, metadata, config = cached_sessions
        loader = EagerSessionCyclingLoader(
            cache_dir=cache_dir,
            metadata=metadata,
            split="train",
            config=config,
        )
        # Should have at least 1 session loaded
        assert len(loader._datasets) > 0
        # Should have pre-computed total samples
        assert loader._total_samples > 0

    def test_batch_shapes(self, cached_sessions):
        """Batches should be (x, y, mask) triples with correct shapes."""
        cache_dir, metadata, config = cached_sessions
        loader = EagerSessionCyclingLoader(
            cache_dir=cache_dir,
            metadata=metadata,
            split="train",
            config=config,
        )

        batch = next(iter(loader))
        x, y, mask = batch
        m_max = metadata["m_max"]
        history_bins = metadata["history_bins"]

        assert x.shape[1] == history_bins  # (batch, T, M)
        assert x.shape[2] == m_max
        assert y.shape[1] == m_max  # (batch, M)
        assert mask.shape[1] == m_max  # (batch, M)

    def test_masks_are_binary(self, cached_sessions):
        """Channel masks should contain only 0s and 1s."""
        cache_dir, metadata, config = cached_sessions
        loader = EagerSessionCyclingLoader(
            cache_dir=cache_dir,
            metadata=metadata,
            split="train",
            config=config,
        )

        for batch in loader:
            _, _, mask = batch
            unique_vals = torch.unique(mask)
            for v in unique_vals:
                assert v.item() in (0.0, 1.0), f"Unexpected mask value: {v}"
            break  # One batch is enough

    def test_len_is_positive(self, cached_sessions):
        """__len__ should return a positive estimated batch count."""
        cache_dir, metadata, config = cached_sessions
        loader = EagerSessionCyclingLoader(
            cache_dir=cache_dir,
            metadata=metadata,
            split="train",
            config=config,
        )
        assert len(loader) > 0

    def test_val_split_no_shuffle(self, cached_sessions):
        """Val split should not shuffle sessions by default."""
        cache_dir, metadata, config = cached_sessions
        loader = EagerSessionCyclingLoader(
            cache_dir=cache_dir,
            metadata=metadata,
            split="val",
            config=config,
        )
        assert loader.shuffle_sessions is False
        assert loader.shuffle_samples is False

    def test_train_split_shuffles(self, cached_sessions):
        """Train split should shuffle sessions by default."""
        cache_dir, metadata, config = cached_sessions
        loader = EagerSessionCyclingLoader(
            cache_dir=cache_dir,
            metadata=metadata,
            split="train",
            config=config,
        )
        assert loader.shuffle_sessions is True
        assert loader.shuffle_samples is True

    def test_exhaustive_iteration(self, cached_sessions):
        """Iterating should yield batches from all sessions."""
        cache_dir, metadata, config = cached_sessions
        loader = EagerSessionCyclingLoader(
            cache_dir=cache_dir,
            metadata=metadata,
            split="train",
            config=config,
            shuffle_sessions=False,  # Deterministic order
        )

        batch_count = 0
        for batch in loader:
            batch_count += 1
            assert len(batch) == 3  # (x, y, mask)

        assert batch_count > 0, "Should yield at least one batch"


# ---------------------------------------------------------------------------
# Eager vs Lazy equivalence tests
# ---------------------------------------------------------------------------

class TestEagerLazyEquivalence:
    """Verify eager and lazy loaders produce equivalent output."""

    def test_same_total_samples(self, cached_sessions):
        """Eager and lazy loaders should have the same total sample count."""
        cache_dir, metadata, config = cached_sessions

        eager = EagerSessionCyclingLoader(
            cache_dir=cache_dir,
            metadata=metadata,
            split="train",
            config=config,
            shuffle_sessions=False,
        )
        lazy = SessionCyclingLoader(
            cache_dir=cache_dir,
            metadata=metadata,
            split="train",
            config=config,
            shuffle_sessions=False,
        )
        assert eager._total_samples == lazy._total_samples

    def test_same_batch_count(self, cached_sessions):
        """Eager and lazy loaders should produce the same number of batches."""
        cache_dir, metadata, config = cached_sessions

        eager = EagerSessionCyclingLoader(
            cache_dir=cache_dir,
            metadata=metadata,
            split="train",
            config=config,
            shuffle_sessions=False,
        )
        lazy = SessionCyclingLoader(
            cache_dir=cache_dir,
            metadata=metadata,
            split="train",
            config=config,
            shuffle_sessions=False,
        )

        eager_batches = sum(1 for _ in eager)
        lazy_batches = sum(1 for _ in lazy)
        assert eager_batches == lazy_batches


# ---------------------------------------------------------------------------
# create_dataloaders factory tests
# ---------------------------------------------------------------------------

class TestCreateDataloaders:
    """Tests for the unified create_dataloaders() factory."""

    def test_lazy_mode_default(self, cached_sessions):
        """Default mode should be 'lazy'."""
        cache_dir, metadata, config = cached_sessions
        loaders = create_dataloaders(cache_dir, metadata, config)
        # Default is lazy, should return SessionCyclingLoader instances
        assert isinstance(loaders["train"], SessionCyclingLoader)

    def test_eager_mode(self, cached_sessions):
        """Setting loader_mode='eager' should return EagerSessionCyclingLoader."""
        cache_dir, metadata, config = cached_sessions
        config["loader_mode"] = "eager"
        loaders = create_dataloaders(cache_dir, metadata, config)
        assert isinstance(loaders["train"], EagerSessionCyclingLoader)
        assert isinstance(loaders["val"], EagerSessionCyclingLoader)
        assert isinstance(loaders["test"], EagerSessionCyclingLoader)

    def test_invalid_mode_raises(self, cached_sessions):
        """Invalid loader_mode should raise ValueError."""
        cache_dir, metadata, config = cached_sessions
        config["loader_mode"] = "invalid"
        with pytest.raises(ValueError, match="Unknown loader_mode"):
            create_dataloaders(cache_dir, metadata, config)

    def test_all_splits_present(self, cached_sessions):
        """Factory should return loaders for all three splits."""
        cache_dir, metadata, config = cached_sessions
        config["loader_mode"] = "eager"
        loaders = create_dataloaders(cache_dir, metadata, config)
        assert set(loaders.keys()) == {"train", "val", "test"}
