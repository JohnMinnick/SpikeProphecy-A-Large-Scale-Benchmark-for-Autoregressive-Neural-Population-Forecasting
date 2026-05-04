"""
Configuration loading and validation utilities.

Provides functions to load YAML configs, merge with defaults, resolve
relative paths, and perform lightweight validation. Every script should
use load_config() as its first step.

Usage:
    from src.utils.config import load_config

    config = load_config("configs/teacher/default.yaml")
"""

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a single YAML file and return its contents as a dictionary.

    Args:
        path: Path to the YAML file (absolute or relative).

    Returns:
        dict: Parsed YAML contents.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Handle empty YAML files
    if data is None:
        data = {}

    logger.debug("Loaded config from %s (%d keys)", path, len(data))
    return data


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries, with override taking precedence.

    Nested dicts are merged recursively. Non-dict values in override
    replace base values. Base is not modified; a new dict is returned.

    Args:
        base: The base (default) dictionary.
        override: The override dictionary whose values take precedence.

    Returns:
        dict: Merged dictionary.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(
    config_path: Union[str, Path],
    defaults_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load a config file, optionally merging with defaults and overrides.

    Precedence (highest to lowest):
    1. overrides (CLI arguments, etc.)
    2. config_path (the primary config file)
    3. defaults_path (base defaults)

    Args:
        config_path: Path to the primary config YAML file.
        defaults_path: Optional path to a defaults YAML file. Values in
            config_path override values in this file.
        overrides: Optional dict of overrides applied on top of
            everything (e.g., from CLI flags).

    Returns:
        dict: Fully resolved configuration dictionary.
    """
    # Start with defaults if provided
    if defaults_path is not None:
        config = load_yaml(defaults_path)
        primary = load_yaml(config_path)
        config = deep_merge(config, primary)
        logger.info(
            "Loaded config: %s (with defaults from %s)",
            config_path,
            defaults_path,
        )
    else:
        config = load_yaml(config_path)
        logger.info("Loaded config: %s", config_path)

    # Apply overrides
    if overrides:
        config = deep_merge(config, overrides)
        logger.debug("Applied %d override keys", len(overrides))

    return config


def resolve_paths(
    config: Dict[str, Any],
    root_dir: Union[str, Path],
    path_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Resolve relative path values in config to absolute paths.

    Only resolves keys listed in path_keys. Paths are resolved relative
    to root_dir (typically the repo root).

    Args:
        config: Configuration dictionary.
        root_dir: Base directory for resolving relative paths.
        path_keys: List of dot-separated keys to resolve (e.g.,
            ["data_dir", "output_dir"]). If None, no resolution is done.

    Returns:
        dict: Config with resolved paths (modified in place and returned).
    """
    if path_keys is None:
        return config

    root = Path(root_dir).resolve()

    for key in path_keys:
        # Support simple top-level keys only for now
        if key in config and isinstance(config[key], str):
            original = config[key]
            resolved = (root / original).resolve()
            config[key] = str(resolved)
            logger.debug("Resolved path '%s': %s → %s", key, original, resolved)

    return config


def save_resolved_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save a fully resolved config to YAML (for experiment logging).

    This is used to write the resolved config.yaml into an experiment
    folder, ensuring no missing defaults.

    Args:
        config: Fully resolved configuration dictionary.
        path: Output path for the YAML file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info("Saved resolved config to %s", path)


def validate_required_keys(
    config: Dict[str, Any],
    required_keys: List[str],
    config_name: str = "config",
) -> None:
    """
    Validate that all required keys are present in a config dict.

    Args:
        config: Configuration dictionary to validate.
        required_keys: List of key names that must be present.
        config_name: Name of the config (for error messages).

    Raises:
        KeyError: If any required key is missing.
    """
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise KeyError(
            f"Missing required keys in {config_name}: {missing}"
        )
