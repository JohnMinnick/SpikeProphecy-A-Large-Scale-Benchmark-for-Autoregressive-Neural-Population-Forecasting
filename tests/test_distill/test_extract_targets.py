"""
Tests for distillation target extraction.

Validates the full pipeline: extract → save → load → validate,
ensuring teacher outputs are correctly cached and round-trip
through serialization without corruption.
"""

import pytest
import torch
import numpy as np

from src.models.teacher import TeacherLSTM
from src.distill.extract_targets import (
    extract_teacher_targets,
    save_distillation_targets,
    load_distillation_targets,
    validate_distillation_targets,
)
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Dimensions for test data
M = 10       # channels
T = 20       # history bins
N_TRAIN = 64 # training samples
N_VAL = 32   # validation samples
BATCH_SIZE = 16


@pytest.fixture
def teacher_model():
    """Create a small teacher model for testing."""
    torch.manual_seed(42)
    model = TeacherLSTM(
        input_size=M,
        hidden_size=32,
        num_layers=1,
        dropout=0.0,
    )
    return model


@pytest.fixture
def device():
    """Resolve test device."""
    return torch.device("cpu")


@pytest.fixture
def dummy_loaders():
    """Create dummy DataLoaders mimicking the real pipeline."""
    torch.manual_seed(42)

    # Create synthetic (input, target) pairs
    train_x = torch.randn(N_TRAIN, T, M).abs()  # non-negative spike counts
    train_y = torch.randn(N_TRAIN, M).abs()
    val_x = torch.randn(N_VAL, T, M).abs()
    val_y = torch.randn(N_VAL, M).abs()

    train_ds = TensorDataset(train_x, train_y)
    val_ds = TensorDataset(val_x, val_y)

    return {
        "train": DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False),
        "val": DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False),
    }


# ---------------------------------------------------------------------------
# Extraction Tests
# ---------------------------------------------------------------------------

class TestExtractTeacherTargets:
    """Tests for the extract_teacher_targets function."""

    def test_returns_all_splits(self, teacher_model, dummy_loaders, device):
        """Extraction should return data for all requested splits."""
        targets = extract_teacher_targets(teacher_model, dummy_loaders, device)
        assert "train" in targets
        assert "val" in targets

    def test_output_shapes(self, teacher_model, dummy_loaders, device):
        """Output tensors should have correct shapes."""
        targets = extract_teacher_targets(teacher_model, dummy_loaders, device)

        # Train split
        assert targets["train"]["inputs"].shape == (N_TRAIN, T, M)
        assert targets["train"]["teacher_rates"].shape == (N_TRAIN, M)
        assert targets["train"]["targets"].shape == (N_TRAIN, M)

        # Val split
        assert targets["val"]["inputs"].shape == (N_VAL, T, M)
        assert targets["val"]["teacher_rates"].shape == (N_VAL, M)
        assert targets["val"]["targets"].shape == (N_VAL, M)

    def test_rates_are_positive(self, teacher_model, dummy_loaders, device):
        """Teacher rates should be non-negative (softplus output)."""
        targets = extract_teacher_targets(teacher_model, dummy_loaders, device)
        for split_name, data in targets.items():
            assert (data["teacher_rates"] >= 0).all(), (
                f"{split_name}: negative rates found"
            )

    def test_rates_are_finite(self, teacher_model, dummy_loaders, device):
        """Teacher rates should contain no NaN or Inf."""
        targets = extract_teacher_targets(teacher_model, dummy_loaders, device)
        for split_name, data in targets.items():
            assert torch.isfinite(data["teacher_rates"]).all(), (
                f"{split_name}: non-finite rates found"
            )

    def test_selective_splits(self, teacher_model, dummy_loaders, device):
        """Should extract only the requested splits."""
        targets = extract_teacher_targets(
            teacher_model, dummy_loaders, device, splits=["val"]
        )
        assert "val" in targets
        assert "train" not in targets

    def test_deterministic(self, teacher_model, dummy_loaders, device):
        """Two extractions with same model should give identical results."""
        torch.manual_seed(0)
        t1 = extract_teacher_targets(teacher_model, dummy_loaders, device)
        torch.manual_seed(0)
        t2 = extract_teacher_targets(teacher_model, dummy_loaders, device)

        torch.testing.assert_close(
            t1["train"]["teacher_rates"],
            t2["train"]["teacher_rates"],
        )

    def test_rates_match_direct_forward(self, teacher_model, dummy_loaders, device):
        """Extracted rates should exactly match a direct model forward pass."""
        targets = extract_teacher_targets(teacher_model, dummy_loaders, device)

        # Manually run the model on the first batch
        teacher_model.eval()
        first_batch_x, _ = next(iter(dummy_loaders["train"]))
        with torch.no_grad():
            expected_rates = teacher_model(first_batch_x.to(device))

        # Compare first BATCH_SIZE samples
        torch.testing.assert_close(
            targets["train"]["teacher_rates"][:BATCH_SIZE],
            expected_rates.cpu(),
        )


# ---------------------------------------------------------------------------
# Save / Load Tests
# ---------------------------------------------------------------------------

class TestSaveAndLoad:
    """Tests for save/load round-trip."""

    def test_save_creates_files(self, teacher_model, dummy_loaders, device, tmp_path):
        """Saving should create one .pt file per split plus metadata."""
        targets = extract_teacher_targets(teacher_model, dummy_loaders, device)
        save_distillation_targets(targets, tmp_path / "distill")

        assert (tmp_path / "distill" / "distill_targets_train.pt").exists()
        assert (tmp_path / "distill" / "distill_targets_val.pt").exists()
        assert (tmp_path / "distill" / "distill_targets_metadata.pt").exists()

    def test_roundtrip_preserves_data(self, teacher_model, dummy_loaders, device, tmp_path):
        """Data should survive save → load without corruption."""
        targets = extract_teacher_targets(teacher_model, dummy_loaders, device)
        save_dir = tmp_path / "distill"
        save_distillation_targets(targets, save_dir)

        # Load back and compare
        loaded = load_distillation_targets(save_dir, split="train")
        torch.testing.assert_close(loaded["inputs"], targets["train"]["inputs"])
        torch.testing.assert_close(loaded["teacher_rates"], targets["train"]["teacher_rates"])
        torch.testing.assert_close(loaded["targets"], targets["train"]["targets"])

    def test_load_missing_split_raises(self, tmp_path):
        """Loading a non-existent split should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_distillation_targets(tmp_path, split="nonexistent")

    def test_file_sizes_reasonable(self, teacher_model, dummy_loaders, device, tmp_path):
        """Saved files should have non-zero size."""
        targets = extract_teacher_targets(teacher_model, dummy_loaders, device)
        save_dir = tmp_path / "distill"
        save_distillation_targets(targets, save_dir)

        train_file = save_dir / "distill_targets_train.pt"
        assert train_file.stat().st_size > 0


# ---------------------------------------------------------------------------
# Validation Tests
# ---------------------------------------------------------------------------

class TestValidateTargets:
    """Tests for the validate_distillation_targets function."""

    def test_valid_targets_pass(self, teacher_model, dummy_loaders, device):
        """Properly extracted targets should pass validation."""
        targets = extract_teacher_targets(teacher_model, dummy_loaders, device)
        stats = validate_distillation_targets(targets)

        assert "train" in stats
        assert "val" in stats
        assert stats["train"]["n_samples"] == N_TRAIN
        assert stats["val"]["n_samples"] == N_VAL

    def test_negative_rates_fails(self, teacher_model, dummy_loaders, device):
        """Targets with negative rates should fail validation."""
        targets = extract_teacher_targets(teacher_model, dummy_loaders, device)
        # Corrupt: insert a negative rate
        targets["train"]["teacher_rates"][0, 0] = -1.0

        with pytest.raises(ValueError, match="negative"):
            validate_distillation_targets(targets)

    def test_nan_rates_fails(self, teacher_model, dummy_loaders, device):
        """Targets with NaN values should fail validation."""
        targets = extract_teacher_targets(teacher_model, dummy_loaders, device)
        targets["train"]["teacher_rates"][0, 0] = float("nan")

        with pytest.raises(ValueError, match="NaN"):
            validate_distillation_targets(targets)

    def test_shape_mismatch_fails(self, teacher_model, dummy_loaders, device):
        """Mismatched shapes should fail validation."""
        targets = extract_teacher_targets(teacher_model, dummy_loaders, device)
        # Corrupt: wrong shape for rates
        targets["train"]["teacher_rates"] = torch.randn(N_TRAIN, M + 5)

        with pytest.raises(ValueError, match="shape"):
            validate_distillation_targets(targets)

    def test_stats_include_rate_range(self, teacher_model, dummy_loaders, device):
        """Validation stats should include rate min/max/mean."""
        targets = extract_teacher_targets(teacher_model, dummy_loaders, device)
        stats = validate_distillation_targets(targets)

        for split_name in ["train", "val"]:
            assert "rate_min" in stats[split_name]
            assert "rate_max" in stats[split_name]
            assert "rate_mean" in stats[split_name]
            assert stats[split_name]["rate_min"] >= 0
