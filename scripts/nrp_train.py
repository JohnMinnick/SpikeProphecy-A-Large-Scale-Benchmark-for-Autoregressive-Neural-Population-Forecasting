"""
NRP Training Entrypoint — Container-side orchestration for NRP jobs.

This script runs inside a Kubernetes pod on the NRP cluster and wraps
the existing train_teacher.py pipeline with:
  1. S3 download (NWB files → data/raw/)
  2. Optional W&B initialization
  3. Training via train_teacher.main()
  4. W&B metric logging (from metrics.json)
  5. S3 upload (experiment folder → S3 outputs)

Usage (inside container):
    python scripts/nrp_train.py

    # Dry-run (skip S3 and W&B, validate config only):
    python scripts/nrp_train.py --dry-run

    # Override configs:
    python scripts/nrp_train.py --teacher-config configs/teacher/nrp_teacher.yaml

Environment variables (set by run_job.sh → jobdefinition.yaml):
    AWS_ACCESS_KEY_ID       — S3 access key
    AWS_SECRET_ACCESS_KEY   — S3 secret key
    WANDB_API_KEY           — Weights & Biases API key (optional)
    S3_DATA_PREFIX          — S3 prefix for input data
    S3_UPLOAD_PREFIX        — S3 prefix for output upload
    INPUT_FILES             — Comma-separated list of NWB files to download
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nrp_train")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for NRP training."""
    parser = argparse.ArgumentParser(
        description="NRP training wrapper: S3 download → train → S3 upload."
    )
    parser.add_argument(
        "--teacher-config",
        type=str,
        default="configs/teacher/nrp_teacher.yaml",
        help="Path to teacher config YAML.",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="configs/data/steinmetz_multi_nrp.yaml",
        help="Path to data config YAML.",
    )
    parser.add_argument(
        "--slug",
        type=str,
        default="nrp_multi_session",
        help="Experiment slug for folder naming.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override max training epochs.",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip S3 and W&B — validate config loading only.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="spike-prophecy",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity (team/user). Uses default if not set.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Download latest checkpoint from S3 and resume training.",
    )
    parser.add_argument(
        "--session-heads",
        action="store_true",
        help="Use session-specific input/output heads instead of shared. "
             "Default is shared heads (better generalization, val_r~0.48).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; load checkpoint (--resume) and run evaluation "
             "with full multi-metric reporting. Saves metrics and exits.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the seed in teacher_config['seed'] (and "
             "data_config['seed'] if present). Used for multi-seed sweeps.",
    )
    return parser.parse_args()


# ====================================================================
# S3 Download
# ====================================================================
def download_nwb_from_s3(dry_run: bool = False) -> None:
    """
    Download NWB files from S3 into data/raw/.

    Reads S3_DATA_PREFIX and INPUT_FILES from environment variables.
    Skips files that already exist locally. In dry-run mode, logs
    what would be downloaded without actually fetching files.
    """
    s3_prefix = os.environ.get("S3_DATA_PREFIX", "<anon>/spike-prophecy/inputs")
    input_files_str = os.environ.get("INPUT_FILES", "")
    data_dir = PROJECT_ROOT / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    if not input_files_str:
        logger.info("INPUT_FILES not set — listing all .nwb files under %s", s3_prefix)
        if dry_run:
            logger.info("[DRY RUN] Would list and download NWB files from S3")
            return
        # Import NRP S3 utilities (available in container)
        nrp_dir = PROJECT_ROOT / "nrp"
        sys.path.insert(0, str(nrp_dir))
        from s3_utils import list_files, download_single_file

        # List all NWB files under the S3 prefix
        all_keys = list_files(s3_prefix)
        nwb_keys = [k for k in all_keys if k.endswith(".nwb")]
        logger.info("Found %d NWB files in S3 under %s", len(nwb_keys), s3_prefix)

        for key in nwb_keys:
            filename = os.path.basename(key)
            local_path = str(data_dir / filename)
            logger.info("Downloading %s → %s", key, local_path)
            download_single_file(key=key, local_path=local_path)
    else:
        # Download specific files listed in INPUT_FILES
        files = [f.strip() for f in input_files_str.split(",") if f.strip()]
        if dry_run:
            logger.info("[DRY RUN] Would download %d files: %s", len(files), files)
            return

        nrp_dir = PROJECT_ROOT / "nrp"
        sys.path.insert(0, str(nrp_dir))
        from s3_utils import download_single_file

        for filename in files:
            s3_key = f"{s3_prefix}/{filename}"
            local_path = str(data_dir / filename)
            logger.info("Downloading %s → %s", s3_key, local_path)
            download_single_file(key=s3_key, local_path=local_path)

    # Verify downloads
    nwb_files = list(data_dir.glob("*.nwb"))
    logger.info("NWB files in data/raw/: %d", len(nwb_files))


def download_checkpoint_from_s3(slug: str) -> Path:
    """
    Download the latest best_model.pt checkpoint from S3 for resume.

    Looks for the checkpoint under the S3 upload prefix at
    <slug>/best_model.pt. Returns the local path if found.

    Args:
        slug: Experiment slug used as the S3 subfolder name.

    Returns:
        Local path to the downloaded checkpoint file.

    Raises:
        FileNotFoundError: If no checkpoint found on S3.
    """
    s3_prefix = os.environ.get(
        "S3_UPLOAD_PREFIX", "<anon>/spike-prophecy/outputs"
    )
    checkpoint_key = f"{s3_prefix}/{slug}/best_model.pt"
    local_path = PROJECT_ROOT / "checkpoints" / "resume_best_model.pt"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    nrp_dir = PROJECT_ROOT / "nrp"
    sys.path.insert(0, str(nrp_dir))
    from s3_utils import download_single_file

    logger.info("Downloading checkpoint from S3: %s", checkpoint_key)
    download_single_file(key=checkpoint_key, local_path=str(local_path))

    if not local_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found on S3: {checkpoint_key}"
        )

    logger.info("Downloaded checkpoint to %s (%.1f MB)",
                local_path, local_path.stat().st_size / 1e6)
    return local_path


def upload_checkpoint_to_s3(checkpoint_path: Path, epoch: int) -> None:
    """
    Upload a checkpoint file to S3.

    Called by the Trainer's checkpoint_callback after best_model.pt
    is saved. This ensures the latest best model is always on S3
    for crash recovery.

    Args:
        checkpoint_path: Local path to the checkpoint .pt file.
        epoch: The epoch number when the checkpoint was saved.
    """
    s3_prefix = os.environ.get(
        "S3_UPLOAD_PREFIX", "<anon>/spike-prophecy/outputs"
    )
    # Get experiment dir name from the checkpoint path
    exp_name = checkpoint_path.parent.name
    s3_sub = exp_name

    nrp_dir = PROJECT_ROOT / "nrp"
    sys.path.insert(0, str(nrp_dir))
    from s3_utils import upload_files

    logger.info(
        "[S3 Checkpoint] Uploading %s → s3://%s/%s/ (epoch %d)",
        checkpoint_path.name, s3_prefix, s3_sub, epoch,
    )
    upload_files(s3_prefix, s3_sub, str(checkpoint_path))
    logger.info("[S3 Checkpoint] Upload complete")


# ====================================================================
# W&B Integration
# ====================================================================
def init_wandb(args: argparse.Namespace, config: dict) -> object:
    """
    Initialize Weights & Biases run if WANDB_API_KEY is set.

    Args:
        args: Parsed command-line arguments.
        config: Combined config dict (teacher + data) to log.

    Returns:
        The wandb run object, or None if W&B is not available.
    """
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if not wandb_key or args.dry_run:
        logger.info("W&B disabled (no API key or dry-run mode)")
        return None

    try:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.slug,
            config=config,
            tags=["nrp", "multi-session", "teacher"],
        )
        logger.info("W&B initialized: %s", run.url)
        return run
    except Exception as e:
        logger.warning("Failed to initialize W&B: %s", e)
        return None


def log_metrics_to_wandb(wandb_run, metrics: dict) -> None:
    """
    Log final training metrics to W&B.

    Extracts key metrics from the experiment's metrics dict and
    logs them as summary values to the W&B run.

    Args:
        wandb_run: Active wandb run object.
        metrics: Metrics dict loaded from metrics.json.
    """
    if wandb_run is None:
        return

    try:
        import wandb

        # Log summary scalars
        summary = {
            "best_val_loss": metrics.get("teacher_best_val_loss"),
            "best_val_pearson_r": metrics.get("teacher_best_val_pearson_r"),
            "n_params": metrics.get("teacher_n_params"),
            "n_epochs": metrics.get("teacher_n_epochs_trained"),
        }

        # Log per-split evaluation results
        split_eval = metrics.get("split_eval", {})
        for _split, split_metrics in split_eval.items():
            for metric_key, metric_val in split_metrics.items():
                summary[metric_key] = metric_val

        wandb.run.summary.update(summary)

        # Log training history as step-wise data if available
        history = metrics.get("history", {})
        if history:
            train_losses = history.get("train_loss", [])
            val_losses = history.get("val_loss", [])
            val_pearson = history.get("val_pearson_r", [])

            for epoch_idx in range(len(train_losses)):
                step_data = {"epoch": epoch_idx + 1}
                step_data["train_loss"] = train_losses[epoch_idx]
                # Val metrics may be sparser than train (val_every_n)
                if epoch_idx < len(val_losses):
                    step_data["val_loss"] = val_losses[epoch_idx]
                if epoch_idx < len(val_pearson):
                    step_data["val_pearson_r"] = val_pearson[epoch_idx]
                wandb.log(step_data)

        logger.info("Metrics logged to W&B")
    except Exception as e:
        logger.warning("Failed to log metrics to W&B: %s", e)


# ====================================================================
# S3 Upload
# ====================================================================
def upload_experiment_to_s3(exp_dir: Path, dry_run: bool = False) -> None:
    """
    Upload the experiment folder to S3.

    Uploads all files in the experiment directory (metrics, plots,
    checkpoints) to the S3 upload prefix.

    Args:
        exp_dir: Path to the experiment directory.
        dry_run: If True, log what would be uploaded without uploading.
    """
    s3_prefix = os.environ.get(
        "S3_UPLOAD_PREFIX", "<anon>/spike-prophecy/outputs"
    )

    if dry_run:
        logger.info("[DRY RUN] Would upload %s → s3://%s/", exp_dir, s3_prefix)
        return

    nrp_dir = PROJECT_ROOT / "nrp"
    sys.path.insert(0, str(nrp_dir))
    from s3_utils import upload_files

    exp_name = exp_dir.name
    upload_count = 0

    for root, _, files in os.walk(exp_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            # Compute relative path within experiment dir
            rel_path = os.path.relpath(root, exp_dir)
            s3_sub = f"{exp_name}/{rel_path}" if rel_path != "." else exp_name

            logger.info("Uploading %s → s3://%s/%s/", filename, s3_prefix, s3_sub)
            upload_files(s3_prefix, s3_sub, file_path)
            upload_count += 1

    logger.info("Uploaded %d files to S3", upload_count)


def upload_experiment_metadata_to_s3(
    exp_dir: Path, dry_run: bool = False
) -> None:
    """
    Upload experiment metadata (config, notes) to S3 before training starts.

    This ensures we always know what a job was configured for, even if
    training crashes on epoch 1. Lightweight — only uploads small text files.

    Args:
        exp_dir: Path to the experiment directory.
        dry_run: If True, log what would be uploaded without uploading.
    """
    s3_prefix = os.environ.get(
        "S3_UPLOAD_PREFIX", "<anon>/spike-prophecy/outputs"
    )
    metadata_files = ["config.yaml", "RUN.md", "notes.md"]

    if dry_run:
        logger.info("[DRY RUN] Would upload metadata from %s", exp_dir)
        return

    nrp_dir = PROJECT_ROOT / "nrp"
    sys.path.insert(0, str(nrp_dir))
    from s3_utils import upload_files

    exp_name = exp_dir.name
    upload_count = 0

    for filename in metadata_files:
        file_path = exp_dir / filename
        if file_path.exists():
            logger.info(
                "[S3 Pre-upload] %s -> s3://%s/%s/",
                filename, s3_prefix, exp_name,
            )
            upload_files(s3_prefix, exp_name, str(file_path))
            upload_count += 1

    logger.info("[S3 Pre-upload] Uploaded %d metadata files", upload_count)


def upload_metrics_to_s3(exp_dir: Path, epoch: int) -> None:
    """
    Upload metrics.json to S3 after a validation pass (incremental).

    Called by the Trainer's metrics_callback after each validation.
    This ensures partial results survive crashes.

    Args:
        exp_dir: Path to the experiment directory.
        epoch: The epoch number when the callback was triggered.
    """
    s3_prefix = os.environ.get(
        "S3_UPLOAD_PREFIX", "<anon>/spike-prophecy/outputs"
    )
    metrics_path = exp_dir / "metrics.json"
    if not metrics_path.exists():
        logger.debug(
            "[S3 Metrics] No metrics.json at epoch %d, skipping", epoch,
        )
        return

    nrp_dir = PROJECT_ROOT / "nrp"
    sys.path.insert(0, str(nrp_dir))
    from s3_utils import upload_files

    exp_name = exp_dir.name
    logger.info(
        "[S3 Metrics] Uploading metrics.json (epoch %d) -> s3://%s/%s/",
        epoch, s3_prefix, exp_name,
    )
    upload_files(s3_prefix, exp_name, str(metrics_path))


# ====================================================================
# Main
# ====================================================================
def main() -> None:
    """Main NRP training pipeline: S3 download → train → W&B log → S3 upload."""
    start_time = time.time()
    args = parse_args()

    logger.info("=" * 60)
    logger.info("NRP TRAINING PIPELINE — SpikeProphecy")
    logger.info("=" * 60)
    logger.info("  Teacher config: %s", args.teacher_config)
    logger.info("  Data config:    %s", args.data_config)
    logger.info("  Slug:           %s", args.slug)
    logger.info("  Dry run:        %s", args.dry_run)
    logger.info("  Project root:   %s", PROJECT_ROOT)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Download NWB data from S3 (skip for IBL — uses ONE API)
    # ------------------------------------------------------------------
    # Quick-peek at data config to check source type before full load
    import yaml
    with open(args.data_config, "r", encoding="utf-8") as _f:
        _data_peek = yaml.safe_load(_f)
    source_type = _data_peek.get("source", {}).get("type", "nwb_multi")

    if source_type == "ibl":
        logger.info("\n[Step 1/5] IBL source — downloading pre-cached arrays from S3")
        # Download pre-cached .npy session files and metadata.json from S3.
        # These were uploaded via scripts/upload_ibl_to_s3.py after local
        # preprocessing with preprocess_and_cache_ibl().
        #
        # Auto-detect S3 prefix from config's ibl.tag field:
        #   "repeated_site" → .../ibl-repeated-site/
        #   "combined"      → .../combined-steinmetz-ibl/
        # Env vars override if explicitly set.
        ibl_tag = _data_peek.get("ibl", {}).get("tag", "repeated_site")
        tag_to_prefix = {
            "repeated_site": "<anon>/spike-prophecy/inputs/ibl-repeated-site",
            "combined": "<anon>/spike-prophecy/inputs/combined-steinmetz-ibl",
        }
        tag_to_cache = {
            "repeated_site": "ibl_repeated_site_cache",
            "combined": "combined_steinmetz_ibl_cache",
        }
        default_prefix = tag_to_prefix.get(
            ibl_tag, f"<anon>/spike-prophecy/inputs/{ibl_tag}"
        )
        default_cache = tag_to_cache.get(
            ibl_tag, f"{ibl_tag}_cache"
        )

        ibl_s3_prefix = os.environ.get("S3_IBL_PREFIX", default_prefix)
        # Local cache dir is configurable for combined runs
        ibl_cache_name = os.environ.get("S3_IBL_CACHE_DIR", default_cache)
        ibl_cache = PROJECT_ROOT / "data" / "processed" / ibl_cache_name
        ibl_cache.mkdir(parents=True, exist_ok=True)

        if not args.dry_run:
            nrp_dir = PROJECT_ROOT / "nrp"
            sys.path.insert(0, str(nrp_dir))
            from s3_utils import list_files, download_single_file

            # List all files under the IBL S3 prefix
            all_keys = list_files(ibl_s3_prefix)
            ibl_keys = [
                k for k in all_keys
                if k.endswith(".npy") or k.endswith(".json")
            ]
            logger.info(
                "Found %d IBL cache files in S3 under %s",
                len(ibl_keys), ibl_s3_prefix,
            )

            for key in ibl_keys:
                filename = os.path.basename(key)
                local_path = str(ibl_cache / filename)
                logger.info("Downloading %s → %s", key, local_path)
                download_single_file(key=key, local_path=local_path)

            # Verify metadata.json was downloaded
            meta_path = ibl_cache / "metadata.json"
            if meta_path.exists():
                import json as _json
                with open(meta_path) as _mf:
                    _ibl_meta = _json.load(_mf)
                logger.info(
                    "IBL cache ready: %d sessions, M_max=%d",
                    _ibl_meta.get("num_sessions", 0),
                    _ibl_meta.get("m_max", 0),
                )
            else:
                logger.warning(
                    "metadata.json not found in IBL cache — "
                    "preprocess_and_cache_ibl() will regenerate"
                )
        else:
            logger.info("[DRY RUN] Would download IBL cache from S3")
    else:
        logger.info("\n[Step 1/5] Downloading NWB data from S3...")
        download_nwb_from_s3(dry_run=args.dry_run)

    # Download existing checkpoint from S3 if resuming
    resume_path = None
    if args.resume and not args.dry_run:
        try:
            resume_path = download_checkpoint_from_s3(args.slug)
            logger.info("Resume checkpoint ready: %s", resume_path)
        except (FileNotFoundError, Exception) as e:
            logger.warning("No checkpoint found for resume: %s", e)
            resume_path = None

    # ------------------------------------------------------------------
    # Step 2: Load configs for W&B
    # ------------------------------------------------------------------
    logger.info("\n[Step 2/5] Loading configs...")
    from src.utils.config import load_config

    teacher_config = load_config(args.teacher_config)
    data_config = load_config(args.data_config)

    # Apply CLI overrides
    if args.epochs is not None:
        teacher_config.setdefault("training", {})["epochs"] = args.epochs
    if args.lr is not None:
        teacher_config.setdefault("training", {})["learning_rate"] = args.lr
    if args.seed is not None:
        teacher_config["seed"] = args.seed
        teacher_config.setdefault("training", {})["seed"] = args.seed
        if isinstance(data_config, dict):
            data_config["seed"] = args.seed
        logger.info("Seed override applied: seed=%d", args.seed)

    combined_config = {"teacher": teacher_config, "data": data_config}

    # ------------------------------------------------------------------
    # Step 2b: Download ceiling weights stats from S3 (if needed)
    # ------------------------------------------------------------------
    cw_cfg = teacher_config.get("loss", {}).get("ceiling_weights", {})
    if cw_cfg.get("enabled", False) and not args.dry_run:
        stats_path = Path(cw_cfg.get(
            "stats_path", "outputs/eval_analysis/per_neuron_stats.json"
        ))
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        if not stats_path.exists():
            logger.info(
                "Downloading per_neuron_stats.json from S3 for ceiling weights..."
            )
            s3_key = "<anon>/spike-prophecy/assets/per_neuron_stats.json"
            nrp_dir = PROJECT_ROOT / "nrp"
            sys.path.insert(0, str(nrp_dir))
            from s3_utils import download_single_file
            download_single_file(key=s3_key, local_path=str(stats_path))
            logger.info(
                "Downloaded ceiling weights stats to %s (%.1f MB)",
                stats_path, stats_path.stat().st_size / 1e6,
            )
        else:
            logger.info("Ceiling weights stats already exist: %s", stats_path)

    # ------------------------------------------------------------------
    # Step 3: Initialize W&B
    # ------------------------------------------------------------------
    logger.info("\n[Step 3/5] Initializing W&B...")
    wandb_run = init_wandb(args, combined_config)

    # ------------------------------------------------------------------
    # Step 4: Run training (delegates to train_teacher.main())
    # ------------------------------------------------------------------
    if args.dry_run:
        # In dry-run mode, validate configs were loaded and exit early.
        # Do NOT run actual training (it will start GPU compute).
        logger.info("\n[Step 4/5] [DRY RUN] Skipping training.")
        logger.info("  Configs validated successfully:")
        logger.info("    Teacher: hidden_size=%s, epochs=%s",
                     teacher_config.get("model", {}).get("hidden_size"),
                     teacher_config.get("training", {}).get("epochs"))
        logger.info("    Data: source=%s",
                     data_config.get("source", {}).get("type"))
        logger.info("\n[Step 5/5] [DRY RUN] Skipping S3 upload and W&B.")
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("DRY RUN COMPLETE — %.1f seconds", elapsed)
        logger.info("=" * 60)
        return

    logger.info("\n[Step 4/5] Running training...")

    # Enable S3 checkpoint callback for NRP (read by train_teacher.py)
    os.environ["NRP_S3_CHECKPOINT"] = "1"
    # Defer W&B finalization: the Trainer normally calls wandb.finish()
    # at end of training, but we need to log final summary metrics first.
    os.environ["NRP_DEFER_WANDB_FINISH"] = "1"

    # Build the argv list that train_teacher.parse_args() expects
    train_argv = [
        "--teacher-config", args.teacher_config,
        "--data-config", args.data_config,
        "--slug", args.slug,
    ]
    if args.epochs is not None:
        train_argv.extend(["--epochs", str(args.epochs)])
    if args.lr is not None:
        train_argv.extend(["--lr", str(args.lr)])
    if args.seed is not None:
        train_argv.extend(["--seed", str(args.seed)])
    # Pass resume checkpoint path if we downloaded one
    if resume_path is not None:
        train_argv.extend(["--resume", str(resume_path)])
    # Pass --session-heads flag through to train_teacher.py
    if args.session_heads:
        train_argv.append("--session-heads")
    # Pass --eval-only flag through to train_teacher.py
    if getattr(args, "eval_only", False):
        train_argv.append("--eval-only")

    # Temporarily replace sys.argv so train_teacher.parse_args() works
    original_argv = sys.argv
    sys.argv = ["train_teacher.py"] + train_argv

    # --- Pre-training S3 upload note ---
    # We cannot upload config before training because train_teacher.main()
    # creates the experiment directory. The try/finally below ensures the
    # full experiment directory (including config) is uploaded to S3 even
    # if training crashes. The incremental metrics_callback handles
    # per-epoch uploads during training.
    exp_root = PROJECT_ROOT / "experiments"

    training_error = None
    try:
        from scripts.train_teacher import main as train_main
        train_main()
    except Exception as e:
        training_error = e
        logger.error("Training failed with error: %s", e, exc_info=True)
    finally:
        sys.argv = original_argv

    # ------------------------------------------------------------------
    # Step 5: Post-training — ALWAYS upload to S3 and log to W&B
    # Runs even if training crashed, so partial results are preserved.
    # ------------------------------------------------------------------
    logger.info("\n[Step 5/5] Post-training upload and logging...")

    # Find the most recent experiment directory
    exp_dirs = sorted(
        [d for d in exp_root.iterdir() if d.is_dir() and d.name != ".gitkeep"],
        key=lambda d: d.stat().st_mtime,
    )

    if exp_dirs:
        latest_exp = exp_dirs[-1]
        logger.info("Latest experiment: %s", latest_exp)

        # Log metrics to W&B (summary + history)
        metrics_file = latest_exp / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, encoding="utf-8") as f:
                metrics = json.load(f)
            log_metrics_to_wandb(wandb_run, metrics)
        else:
            logger.warning(
                "No metrics.json found in %s — training may have "
                "crashed before saving metrics", latest_exp,
            )

        # Upload FULL experiment folder to S3 (crash-safe)
        upload_experiment_to_s3(latest_exp, dry_run=args.dry_run)
    else:
        logger.warning("No experiment directories found!")

    # Finish W&B run (deferred from Trainer)
    if wandb_run is not None:
        try:
            import wandb
            wandb.finish()
            logger.info("W&B run finished")
        except Exception as e:
            logger.warning("Failed to finish W&B run: %s", e)

    elapsed = time.time() - start_time

    # Re-raise training error after cleanup is done
    if training_error is not None:
        logger.error(
            "NRP PIPELINE FAILED after %.1f minutes", elapsed / 60,
        )
        raise training_error

    logger.info("=" * 60)
    logger.info("NRP PIPELINE COMPLETE — %.1f minutes", elapsed / 60)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
