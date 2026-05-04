"""
Tests for src/utils/experiment.py

Tests experiment folder creation, immutability enforcement,
metrics saving, and experiment listing.
"""

import json
from pathlib import Path

import pytest
import yaml

from src.utils.experiment import (
    create_experiment,
    list_experiments,
    save_metrics,
)


@pytest.fixture
def exp_root(tmp_path):
    """Provide a temporary experiments directory."""
    return tmp_path / "experiments"


class TestCreateExperiment:
    """Tests for create_experiment()."""

    def test_creates_folder_with_correct_name(self, exp_root):
        """Folder name should match YYYY-MM-DD_slug pattern."""
        exp_dir = create_experiment(
            slug="test_run",
            config={"lr": 0.001},
            command="python train.py",
            experiments_dir=exp_root,
        )
        assert exp_dir.exists()
        assert exp_dir.name.endswith("_test_run")
        # Verify date prefix is YYYY-MM-DD format
        date_part = exp_dir.name.split("_")[0]
        parts = date_part.split("-")
        assert len(parts) == 3  # year, month, day
        assert len(parts[0]) == 4  # 4-digit year

    def test_creates_all_required_files(self, exp_root):
        """Should create config.yaml, RUN.md, notes.md, metrics.json, plots/."""
        exp_dir = create_experiment(
            slug="file_check",
            config={"model": "lstm"},
            command="python train.py",
            experiments_dir=exp_root,
        )
        assert (exp_dir / "config.yaml").is_file()
        assert (exp_dir / "RUN.md").is_file()
        assert (exp_dir / "notes.md").is_file()
        assert (exp_dir / "metrics.json").is_file()
        assert (exp_dir / "plots").is_dir()

    def test_config_yaml_contains_correct_values(self, exp_root):
        """Saved config.yaml should match the input config."""
        config = {"architecture": "lstm", "hidden_size": 128, "seed": 42}
        exp_dir = create_experiment(
            slug="config_check",
            config=config,
            command="python train.py",
            experiments_dir=exp_root,
        )
        saved = yaml.safe_load(open(exp_dir / "config.yaml"))
        assert saved["architecture"] == "lstm"
        assert saved["hidden_size"] == 128
        assert saved["seed"] == 42

    def test_run_md_contains_command(self, exp_root):
        """RUN.md should contain the exact command used."""
        command = "python scripts/train_teacher.py --config default.yaml"
        exp_dir = create_experiment(
            slug="cmd_check",
            config={},
            command=command,
            experiments_dir=exp_root,
        )
        run_md = (exp_dir / "RUN.md").read_text()
        assert command in run_md

    def test_immutability_raises_on_duplicate(self, exp_root):
        """Creating same slug on same day should raise FileExistsError."""
        create_experiment(
            slug="unique_run",
            config={},
            command="python train.py",
            experiments_dir=exp_root,
        )
        with pytest.raises(FileExistsError, match="already exists"):
            create_experiment(
                slug="unique_run",
                config={},
                command="python train.py",
                experiments_dir=exp_root,
            )

    def test_metrics_json_starts_empty(self, exp_root):
        """metrics.json should start as an empty dict."""
        exp_dir = create_experiment(
            slug="metrics_init",
            config={},
            command="python train.py",
            experiments_dir=exp_root,
        )
        with open(exp_dir / "metrics.json") as f:
            metrics = json.load(f)
        assert metrics == {}

    def test_custom_notes(self, exp_root):
        """Custom notes should appear in notes.md."""
        exp_dir = create_experiment(
            slug="noted",
            config={},
            command="python train.py",
            experiments_dir=exp_root,
            notes="Increased learning rate to 0.01",
        )
        notes = (exp_dir / "notes.md").read_text()
        assert "Increased learning rate to 0.01" in notes


class TestSaveMetrics:
    """Tests for save_metrics()."""

    def test_saves_metrics(self, exp_root):
        """Should write metrics to metrics.json."""
        exp_dir = create_experiment(
            slug="save_met",
            config={},
            command="test",
            experiments_dir=exp_root,
        )
        save_metrics(exp_dir, {"loss": 0.5, "accuracy": 0.95})
        with open(exp_dir / "metrics.json") as f:
            metrics = json.load(f)
        assert metrics["loss"] == 0.5
        assert metrics["accuracy"] == 0.95

    def test_appends_metrics(self, exp_root):
        """Append mode should merge with existing metrics."""
        exp_dir = create_experiment(
            slug="append_met",
            config={},
            command="test",
            experiments_dir=exp_root,
        )
        save_metrics(exp_dir, {"epoch_1_loss": 1.0})
        save_metrics(exp_dir, {"epoch_2_loss": 0.5}, append=True)
        with open(exp_dir / "metrics.json") as f:
            metrics = json.load(f)
        assert metrics["epoch_1_loss"] == 1.0
        assert metrics["epoch_2_loss"] == 0.5

    def test_overwrite_metrics(self, exp_root):
        """Overwrite mode should replace all existing metrics."""
        exp_dir = create_experiment(
            slug="overwrite_met",
            config={},
            command="test",
            experiments_dir=exp_root,
        )
        save_metrics(exp_dir, {"old": 1.0})
        save_metrics(exp_dir, {"new": 2.0}, append=False)
        with open(exp_dir / "metrics.json") as f:
            metrics = json.load(f)
        assert "old" not in metrics
        assert metrics["new"] == 2.0


class TestListExperiments:
    """Tests for list_experiments()."""

    def test_empty_dir_returns_empty(self, tmp_path):
        """Empty experiments dir should return empty list."""
        result = list_experiments(tmp_path / "nonexistent")
        assert result == []

    def test_lists_experiment_folders(self, exp_root):
        """Should list created experiment folders."""
        create_experiment("run_a", {}, "test", experiments_dir=exp_root)
        create_experiment("run_b", {}, "test", experiments_dir=exp_root)
        folders = list_experiments(exp_root)
        assert len(folders) == 2
        # Folder names are YYYY-MM-DD_slug, check slugs are present
        names = [f.name for f in folders]
        assert any(n.endswith("_run_a") for n in names)
        assert any(n.endswith("_run_b") for n in names)
