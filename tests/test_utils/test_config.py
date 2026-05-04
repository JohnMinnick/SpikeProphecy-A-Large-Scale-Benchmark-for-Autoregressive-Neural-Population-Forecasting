"""
Tests for src/utils/config.py

Tests YAML loading, deep merge, config validation, path resolution,
and config saving. Uses concrete expected values for validation.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.utils.config import (
    deep_merge,
    load_config,
    load_yaml,
    resolve_paths,
    save_resolved_config,
    validate_required_keys,
)


@pytest.fixture
def tmp_yaml(tmp_path):
    """Create a temporary YAML file for testing."""
    config = {
        "model": {"type": "lstm", "hidden": 128},
        "optimizer": {"lr": 0.001},
        "seed": 42,
    }
    path = tmp_path / "test_config.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


@pytest.fixture
def tmp_defaults_yaml(tmp_path):
    """Create a defaults YAML file for testing merge behavior."""
    defaults = {
        "model": {"type": "lstm", "hidden": 64, "dropout": 0.1},
        "optimizer": {"lr": 0.01, "weight_decay": 1e-5},
        "seed": 0,
        "batch_size": 32,
    }
    path = tmp_path / "defaults.yaml"
    with open(path, "w") as f:
        yaml.dump(defaults, f)
    return path


class TestLoadYaml:
    """Tests for load_yaml()."""

    def test_loads_yaml_file(self, tmp_yaml):
        """Should load YAML and return a dict with correct values."""
        data = load_yaml(tmp_yaml)
        assert data["seed"] == 42
        assert data["model"]["type"] == "lstm"
        assert data["model"]["hidden"] == 128

    def test_missing_file_raises(self):
        """Non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_yaml("nonexistent_config.yaml")

    def test_empty_yaml_returns_empty_dict(self, tmp_path):
        """Empty YAML file should return empty dict, not None."""
        path = tmp_path / "empty.yaml"
        path.write_text("")
        data = load_yaml(path)
        assert data == {}


class TestDeepMerge:
    """Tests for deep_merge()."""

    def test_flat_merge(self):
        """Non-nested override should replace base values."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        """Nested dicts should be merged recursively."""
        base = {"model": {"type": "lstm", "hidden": 64, "dropout": 0.1}}
        override = {"model": {"hidden": 128}}
        result = deep_merge(base, override)
        # hidden was overridden
        assert result["model"]["hidden"] == 128
        # type and dropout preserved from base
        assert result["model"]["type"] == "lstm"
        assert result["model"]["dropout"] == 0.1

    def test_base_not_mutated(self):
        """Original base dict should not be modified."""
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        deep_merge(base, override)
        assert base["a"]["b"] == 1  # Original unchanged

    def test_override_non_dict_replaces_dict(self):
        """A non-dict override should replace a dict base value."""
        base = {"a": {"nested": True}}
        override = {"a": "flat_string"}
        result = deep_merge(base, override)
        assert result["a"] == "flat_string"


class TestLoadConfig:
    """Tests for load_config()."""

    def test_load_simple_config(self, tmp_yaml):
        """Should load the primary config file."""
        config = load_config(tmp_yaml)
        assert config["seed"] == 42
        assert config["model"]["hidden"] == 128

    def test_load_with_defaults(self, tmp_yaml, tmp_defaults_yaml):
        """Primary config should override defaults where specified."""
        config = load_config(tmp_yaml, defaults_path=tmp_defaults_yaml)
        # Overridden by primary
        assert config["model"]["hidden"] == 128
        assert config["optimizer"]["lr"] == 0.001
        assert config["seed"] == 42
        # Preserved from defaults (not in primary)
        assert config["model"]["dropout"] == 0.1
        assert config["optimizer"]["weight_decay"] == 1e-5
        assert config["batch_size"] == 32

    def test_load_with_overrides(self, tmp_yaml):
        """CLI overrides should take highest precedence."""
        config = load_config(tmp_yaml, overrides={"seed": 999})
        assert config["seed"] == 999

    def test_load_actual_project_config(self):
        """Smoke test: load the actual project data config."""
        config = load_config("configs/data/default.yaml")
        assert config["bin_width_ms"] == 10
        assert config["history_bins"] == 50
        assert config["forecast_horizon"] == 1
        assert config["source"]["type"] == "spikeinterface"


class TestResolvePaths:
    """Tests for resolve_paths()."""

    def test_resolves_relative_path(self, tmp_path):
        """Relative path should be resolved to absolute."""
        config = {"data_dir": "data/raw"}
        result = resolve_paths(config, tmp_path, path_keys=["data_dir"])
        resolved = result["data_dir"]
        assert Path(resolved).is_absolute()
        assert resolved.endswith("data\\raw") or resolved.endswith("data/raw")

    def test_no_path_keys_returns_unchanged(self):
        """No path_keys should return config unchanged."""
        config = {"data_dir": "data/raw"}
        result = resolve_paths(config, "/tmp", path_keys=None)
        assert result["data_dir"] == "data/raw"


class TestValidateRequiredKeys:
    """Tests for validate_required_keys()."""

    def test_valid_config_passes(self):
        """Config with all required keys should not raise."""
        config = {"a": 1, "b": 2, "c": 3}
        validate_required_keys(config, ["a", "b"])  # Should not raise

    def test_missing_key_raises(self):
        """Missing required key should raise KeyError with the key name."""
        config = {"a": 1}
        with pytest.raises(KeyError, match="missing_key"):
            validate_required_keys(config, ["a", "missing_key"])


class TestSaveResolvedConfig:
    """Tests for save_resolved_config()."""

    def test_saves_and_reloads(self, tmp_path):
        """Saved config should be reloadable with identical values."""
        config = {"model": {"type": "lstm"}, "seed": 42}
        path = tmp_path / "saved.yaml"
        save_resolved_config(config, path)

        # Reload and verify
        reloaded = load_yaml(path)
        assert reloaded == config

    def test_creates_parent_dirs(self, tmp_path):
        """Should create parent directories if they don't exist."""
        path = tmp_path / "nested" / "dir" / "config.yaml"
        save_resolved_config({"key": "value"}, path)
        assert path.exists()
