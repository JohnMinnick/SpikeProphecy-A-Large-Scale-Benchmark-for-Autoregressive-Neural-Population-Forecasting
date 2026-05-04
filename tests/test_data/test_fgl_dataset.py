"""
Tests for src/data/fgl_dataset.py

Validates the FGL temporal offset mechanism and dataset construction
with synthetic data where offsets can be verified analytically.
"""

import numpy as np
import pytest
import torch

from src.data.fgl_dataset import FGLDataset


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_series():
    """
    Create a simple deterministic time-series for testing.

    Values are set to the time index for easy offset verification:
    data[t, :] = t for all channels.
    """
    T_total = 50
    M = 5
    data = torch.arange(T_total, dtype=torch.float32).unsqueeze(1).expand(-1, M)
    return data  # (50, 5) — each row = [t, t, t, t, t]


@pytest.fixture
def poisson_series():
    """Realistic Poisson-distributed spike-count time series."""
    rng = np.random.RandomState(42)
    T_total = 200
    M = 10
    data = torch.tensor(
        rng.poisson(lam=2.0, size=(T_total, M)),
        dtype=torch.float32,
    )
    return data


# =============================================================================
# Constructor tests
# =============================================================================

class TestFGLDatasetConstruction:
    """Tests for FGLDataset construction and validation."""

    def test_basic_creation(self, simple_series):
        """Should create dataset with correct sample count."""
        ds = FGLDataset(simple_series, history_bins=10, K=5)
        # num_samples = T_total - K - T = 50 - 5 - 10 = 35
        assert len(ds) == 35

    def test_sample_reduction_by_K(self, simple_series):
        """Increasing K should reduce the number of samples."""
        ds_k1 = FGLDataset(simple_series, history_bins=10, K=1)
        ds_k5 = FGLDataset(simple_series, history_bins=10, K=5)
        ds_k10 = FGLDataset(simple_series, history_bins=10, K=10)
        assert len(ds_k1) > len(ds_k5) > len(ds_k10)

    def test_too_large_K_raises(self, simple_series):
        """K + T >= T_total should raise ValueError."""
        with pytest.raises(ValueError, match="Not enough data"):
            FGLDataset(simple_series, history_bins=10, K=41)

    def test_zero_K_allowed(self, simple_series):
        """K=0 is a degenerate case (teacher = student), should still work."""
        ds = FGLDataset(simple_series, history_bins=10, K=0)
        assert len(ds) == 40  # Same as standard dataset


# =============================================================================
# Indexing and offset tests
# =============================================================================

class TestFGLDatasetOffset:
    """Tests for temporal offset correctness."""

    def test_student_window_position(self, simple_series):
        """Student window should start at idx and have T bins."""
        T = 10
        K = 5
        ds = FGLDataset(simple_series, history_bins=T, K=K)

        x_student, x_teacher, y_target = ds[0]
        # Student: bins [0 .. 9]
        assert x_student.shape == (T, 5)
        assert x_student[0, 0].item() == 0.0
        assert x_student[-1, 0].item() == 9.0

    def test_teacher_window_offset(self, simple_series):
        """Teacher window should be offset by K bins from student."""
        T = 10
        K = 5
        ds = FGLDataset(simple_series, history_bins=T, K=K)

        x_student, x_teacher, y_target = ds[0]
        # Teacher: bins [K .. K+T-1] = [5 .. 14]
        assert x_teacher[0, 0].item() == 5.0   # = K
        assert x_teacher[-1, 0].item() == 14.0  # = K + T - 1

    def test_target_position(self, simple_series):
        """Target should be at bin idx + K + T."""
        T = 10
        K = 5
        ds = FGLDataset(simple_series, history_bins=T, K=K)

        x_student, x_teacher, y_target = ds[0]
        # Target: bin K + T = 5 + 10 = 15
        assert y_target[0].item() == 15.0

    def test_student_teacher_gap_is_K(self, simple_series):
        """The start of teacher window should be exactly K bins after student start."""
        T = 10
        K = 7
        ds = FGLDataset(simple_series, history_bins=T, K=K)

        for idx in [0, 3, 10]:
            x_s, x_t, _ = ds[idx]
            gap = x_t[0, 0].item() - x_s[0, 0].item()
            assert gap == K, f"Gap at idx={idx} is {gap}, expected {K}"

    def test_multiple_indices_consistent(self, simple_series):
        """Verify offsets hold across multiple indices."""
        T = 10
        K = 5
        ds = FGLDataset(simple_series, history_bins=T, K=K)

        for idx in range(0, len(ds), 5):
            x_s, x_t, y = ds[idx]
            # Student starts at idx
            assert x_s[0, 0].item() == idx
            # Teacher starts at idx + K
            assert x_t[0, 0].item() == idx + K
            # Target at idx + K + T
            assert y[0].item() == idx + K + T

    def test_out_of_bounds_raises(self, simple_series):
        """Accessing past the end should raise IndexError."""
        ds = FGLDataset(simple_series, history_bins=10, K=5)
        with pytest.raises(IndexError):
            ds[len(ds)]


# =============================================================================
# Output shape tests
# =============================================================================

class TestFGLDatasetShapes:
    """Tests for output tensor shapes."""

    def test_output_shapes(self, poisson_series):
        """All outputs should have correct shapes."""
        T = 15
        K = 5
        M = poisson_series.shape[1]
        ds = FGLDataset(poisson_series, history_bins=T, K=K)

        x_s, x_t, y = ds[0]
        assert x_s.shape == (T, M)
        assert x_t.shape == (T, M)
        assert y.shape == (M,)

    def test_num_channels_with_concat(self, poisson_series):
        """num_channels does not trim y_target — uses full M for ConcatDataset compat."""
        # When mixing sessions with different M via ConcatDataset,
        # y_target must be uniform (all M_max channels).  Loss masking
        # handles the zero-padded channels downstream.
        ds = FGLDataset(
            poisson_series, history_bins=10, K=5, num_channels=5,
        )
        x_s, x_t, y = ds[0]
        assert y.shape == (10,)          # Full data width (not trimmed)
        assert x_s.shape[1] == 10       # Full input (includes features)
