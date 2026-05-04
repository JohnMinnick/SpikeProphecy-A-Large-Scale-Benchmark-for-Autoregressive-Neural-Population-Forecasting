"""
Experiment folder scaffolding and logging utilities.

Manages creation of experiment folders per project rules §6:
    experiments/YYYY-MM-DD_<slug>/
        config.yaml      — fully resolved config
        RUN.md           — exact command, environment, git hash
        metrics.json     — machine-readable results (created later)
        notes.md         — what changed and why (created later)
        plots/           — exported figures (created later)

Usage:
    from src.utils.experiment import create_experiment

    exp_dir = create_experiment(
        slug="teacher_baseline_v1",
        config=resolved_config,
        command="python scripts/train_teacher.py --config configs/teacher/default.yaml",
    )
"""

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)

# Default experiments root relative to repo root
DEFAULT_EXPERIMENTS_DIR = "experiments"


def _get_git_hash() -> str:
    """
    Get the current git commit hash (short form).

    Returns:
        str: Short git hash, or "unknown" if git is not available.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("Could not determine git hash")
        return "unknown"


def _get_git_dirty() -> bool:
    """
    Check if the git working tree has uncommitted changes.

    Returns:
        bool: True if there are uncommitted changes.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode != 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def create_experiment(
    slug: str,
    config: Dict[str, Any],
    command: str,
    experiments_dir: Union[str, Path] = DEFAULT_EXPERIMENTS_DIR,
    notes: Optional[str] = None,
    env_notes: Optional[str] = None,
) -> Path:
    """
    Create a new experiment folder with all required scaffolding.

    The folder name is: YYYY-MM-DD_<slug>/
    Raises an error if the folder already exists (experiments are immutable).

    Args:
        slug: Short descriptive name for the experiment (snake_case).
        config: Fully resolved configuration dictionary.
        command: The exact command used to launch this experiment.
        experiments_dir: Root directory for experiments (default: "experiments/").
        notes: Optional initial notes for notes.md (what changed and why).
        env_notes: Optional environment notes (e.g., machine name, GPU).

    Returns:
        Path: Absolute path to the created experiment folder.

    Raises:
        FileExistsError: If the experiment folder already exists.
    """
    # Build folder name: YYYY-MM-DD_slug
    date_str = datetime.now().strftime("%Y-%m-%d")
    folder_name = f"{date_str}_{slug}"
    exp_dir = Path(experiments_dir).resolve() / folder_name

    # Immutability check: do not overwrite existing experiments
    if exp_dir.exists():
        raise FileExistsError(
            f"Experiment folder already exists: {exp_dir}. "
            "Per project rules, experiments are immutable. "
            "Use a different slug or wait until tomorrow."
        )

    # Create directory structure
    exp_dir.mkdir(parents=True)
    (exp_dir / "plots").mkdir()
    logger.info("Created experiment folder: %s", exp_dir)

    # --- config.yaml (fully resolved) ---
    config_path = exp_dir / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.debug("Wrote resolved config to %s", config_path)

    # --- RUN.md ---
    git_hash = _get_git_hash()
    git_dirty = " (dirty)" if _get_git_dirty() else ""
    run_md = f"""# Run Record

## Command
```
{command}
```

## Git
- Commit: `{git_hash}{git_dirty}`

## Environment
{env_notes or "Not specified."}

## Date
{date_str}
"""
    run_path = exp_dir / "RUN.md"
    with open(run_path, "w", encoding="utf-8") as f:
        f.write(run_md)
    logger.debug("Wrote RUN.md to %s", run_path)

    # --- notes.md (initial) ---
    notes_content = f"# Experiment Notes: {slug}\n\n"
    if notes:
        notes_content += f"{notes}\n"
    else:
        notes_content += "*(Add notes about what changed vs. previous run and why.)*\n"

    notes_path = exp_dir / "notes.md"
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write(notes_content)

    # --- metrics.json (empty placeholder) ---
    metrics_path = exp_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({}, f)

    logger.info("Experiment '%s' scaffolded at %s", slug, exp_dir)
    return exp_dir


def save_metrics(
    exp_dir: Union[str, Path],
    metrics: Dict[str, Any],
    append: bool = True,
) -> None:
    """
    Save or update metrics.json in an experiment folder.

    Args:
        exp_dir: Path to the experiment folder.
        metrics: Dictionary of metric name → value pairs.
        append: If True, merge with existing metrics. If False, overwrite.
    """
    metrics_path = Path(exp_dir) / "metrics.json"

    if append and metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing.update(metrics)
        metrics = existing

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved %d metrics to %s", len(metrics), metrics_path)


def list_experiments(
    experiments_dir: Union[str, Path] = DEFAULT_EXPERIMENTS_DIR,
) -> List[Path]:
    """
    List all experiment folders, sorted by date (newest first).

    Args:
        experiments_dir: Root directory for experiments.

    Returns:
        list[Path]: List of experiment folder paths, newest first.
    """
    exp_root = Path(experiments_dir)
    if not exp_root.exists():
        return []

    # Filter for directories matching the YYYY-MM-DD_ pattern
    folders = sorted(
        [d for d in exp_root.iterdir() if d.is_dir() and d.name[0:4].isdigit()],
        reverse=True,
    )
    return folders
