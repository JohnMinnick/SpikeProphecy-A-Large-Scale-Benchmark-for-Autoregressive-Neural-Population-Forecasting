"""
Train a Student SNN via distillation from a pretrained teacher on NRP.

Loads a pretrained teacher checkpoint (from S3 or local), preprocesses
multi-session NWB data, and trains a StudentSNN using DistillTrainer
with online teacher inference per batch.

The teacher is frozen and used to generate soft targets on-the-fly,
avoiding the need to pre-extract and store large .pt target files.

Usage:
    # Local smoke test:
    python scripts/train_distill.py \
        --teacher-checkpoint experiments/<exp>/best_model.pt \
        --data-config configs/data/steinmetz_multi_nrp_50ms.yaml \
        --teacher-config configs/teacher/nrp_teacher_lru_v2.yaml \
        --student-config configs/student/distill_nrp.yaml \
        --slug distill-smoke --epochs 3

    # NRP container:
    python scripts/train_distill.py \
        --teacher-s3-slug 2026-03-01_lru-v2-temporal \
        --data-config configs/data/steinmetz_multi_nrp_50ms.yaml \
        --teacher-config configs/teacher/nrp_teacher_lru_v2.yaml \
        --student-config configs/student/distill_nrp.yaml \
        --slug distill-snn-v1
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.distill_dataset import DistillCollator
from src.data.multi_session_loader import preprocess_and_cache, create_dataloaders
from src.distill.distill_trainer import DistillTrainer
from src.distill.loss import DistillationLoss
from src.models.student import StudentSNN
from src.models.teacher import create_teacher_model
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.experiment import create_experiment
from src.utils.seed import seed_everything

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_distill")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for SNN distillation training."""
    parser = argparse.ArgumentParser(
        description="Train Student SNN via distillation from pretrained teacher."
    )
    parser.add_argument(
        "--teacher-config", type=str, required=True,
        help="Path to teacher model config YAML.",
    )
    parser.add_argument(
        "--data-config", type=str, required=True,
        help="Path to data config YAML.",
    )
    parser.add_argument(
        "--student-config", type=str, required=True,
        help="Path to student model config YAML.",
    )
    parser.add_argument(
        "--teacher-checkpoint", type=str, default="",
        help="Local path to pretrained teacher checkpoint (.pt).",
    )
    parser.add_argument(
        "--teacher-s3-slug", type=str, default="",
        help="S3 experiment slug to download teacher checkpoint from.",
    )
    parser.add_argument(
        "--slug", type=str, default="distill-snn",
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
        "--dry-run", action="store_true",
        help="Validate config and checkpoint loading only.",
    )
    return parser.parse_args()


def main() -> None:
    """Main SNN distillation pipeline."""
    start_time = time.time()
    args = parse_args()

    logger.info("=" * 60)
    logger.info("SNN DISTILLATION TRAINING — SpikeProphecy (Phase 4)")
    logger.info("=" * 60)
    logger.info("  Teacher config:     %s", args.teacher_config)
    logger.info("  Student config:     %s", args.student_config)
    logger.info("  Data config:        %s", args.data_config)
    logger.info("  Teacher checkpoint: %s", args.teacher_checkpoint or "(from S3)")
    logger.info("  Slug:               %s", args.slug)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load configs
    # ------------------------------------------------------------------
    teacher_config = load_config(args.teacher_config)
    data_config = load_config(args.data_config)
    student_config = load_config(args.student_config)

    # Apply CLI overrides to student config
    if args.epochs is not None:
        student_config.setdefault("training", {})["epochs"] = args.epochs
    if args.lr is not None:
        student_config.setdefault("training", {})["learning_rate"] = args.lr

    # ------------------------------------------------------------------
    # 2. Download data and teacher checkpoint from S3 if NRP mode
    # ------------------------------------------------------------------
    teacher_checkpoint_path = args.teacher_checkpoint

    if os.environ.get("AWS_ACCESS_KEY_ID"):
        logger.info("NRP mode — downloading NWB data from S3...")
        try:
            from scripts.nrp_train import download_nwb_from_s3
            download_nwb_from_s3()
        except ImportError:
            logger.warning("Could not import nrp_train — skipping S3 download")

        # Download teacher checkpoint from S3 if slug provided
        if args.teacher_s3_slug:
            logger.info(
                "Downloading teacher checkpoint from S3: %s",
                args.teacher_s3_slug,
            )
            try:
                from scripts.nrp_train import download_checkpoint_from_s3
                teacher_checkpoint_path = str(
                    download_checkpoint_from_s3(args.teacher_s3_slug)
                )
                logger.info("Teacher checkpoint: %s", teacher_checkpoint_path)
            except Exception as e:
                logger.error("Failed to download teacher checkpoint: %s", e)
                sys.exit(1)

    if not teacher_checkpoint_path:
        logger.error(
            "No teacher checkpoint specified. Use --teacher-checkpoint "
            "or --teacher-s3-slug."
        )
        sys.exit(1)

    if args.dry_run:
        logger.info("[DRY RUN] Configs validated.")
        if teacher_checkpoint_path and Path(teacher_checkpoint_path).exists():
            logger.info("[DRY RUN] Checkpoint exists: %s", teacher_checkpoint_path)
        else:
            logger.info("[DRY RUN] Checkpoint path: %s (not checked)", teacher_checkpoint_path)
        logger.info("[DRY RUN] Exiting.")
        return

    # ------------------------------------------------------------------
    # 3. Set seed for reproducibility
    # ------------------------------------------------------------------
    seed = student_config.get("training", {}).get("seed", 42)
    seed_everything(seed)

    # ------------------------------------------------------------------
    # 4. Preprocess data and create base DataLoaders
    # ------------------------------------------------------------------
    logger.info("Preprocessing multi-session data...")
    cache_dir, multi_meta = preprocess_and_cache(data_config)
    m_max = multi_meta["m_max"]
    logger.info("Data ready: M_max=%d", m_max)

    batch_size = student_config.get("training", {}).get("batch_size", 512)
    base_loaders = create_dataloaders(cache_dir, multi_meta, data_config)

    # ------------------------------------------------------------------
    # 5. Resolve device
    # ------------------------------------------------------------------
    device = resolve_device()
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    # 6. Load pretrained teacher (frozen)
    # ------------------------------------------------------------------
    logger.info("Loading pretrained teacher from %s...", teacher_checkpoint_path)
    model_config = teacher_config.get("model", {})

    # create_teacher_model expects the full teacher config dict
    # (it extracts model.architecture internally for dispatch)
    teacher = create_teacher_model(
        config=teacher_config,
        input_size=m_max,
    )
    checkpoint = torch.load(
        teacher_checkpoint_path, map_location=device, weights_only=True,
    )
    # Handle both full checkpoint dicts and raw state dicts
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    teacher.load_state_dict(state_dict)
    teacher.to(device)
    teacher.eval()

    # Freeze teacher parameters
    for param in teacher.parameters():
        param.requires_grad_(False)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    logger.info("Teacher loaded and frozen: %d params", teacher_params)

    # ------------------------------------------------------------------
    # 7. Create DistillCollator for online teacher inference
    # ------------------------------------------------------------------
    # 7. Create distill loader wrapper for online teacher inference
    # ------------------------------------------------------------------
    # SessionCyclingLoader yields pre-batched (x, y, mask) tuples.
    # We wrap it to run frozen teacher on each batch and produce
    # (x, y, teacher_rates) triplets that DistillTrainer expects.
    class DistillLoaderWrapper:
        """Wraps a base loader to add teacher predictions per-batch."""

        def __init__(self, base_loader, teacher_model, dev, out_channels):
            self.base_loader = base_loader
            self.teacher_model = teacher_model
            self.dev = dev
            self.out_channels = out_channels

        def __iter__(self):
            for batch in self.base_loader:
                x, y = batch[0], batch[1]
                # Run frozen teacher on GPU
                with torch.no_grad():
                    x_dev = x.to(self.dev)
                    teacher_out = self.teacher_model(x_dev)
                    if isinstance(teacher_out, dict):
                        teacher_rates = teacher_out["rates"]
                    else:
                        teacher_rates = teacher_out
                    # Slice to output_channels
                    if self.out_channels is not None:
                        teacher_rates = teacher_rates[:, :self.out_channels]
                        y = y[:, :self.out_channels]
                # Return CPU tensors — DistillTrainer handles device
                yield x.cpu(), y.cpu(), teacher_rates.cpu()

        def __len__(self):
            return len(self.base_loader)

    distill_loaders = {}
    for split_name, base_loader in base_loaders.items():
        distill_loaders[split_name] = DistillLoaderWrapper(
            base_loader, teacher, device, m_max,
        )
        logger.info(
            "Distill %s loader: wrapped with online teacher inference",
            split_name,
        )

    # ------------------------------------------------------------------
    # 8. Create Student SNN
    # ------------------------------------------------------------------
    logger.info("Creating StudentSNN (random init)...")
    student_model_cfg = student_config.get("model", {})

    student = StudentSNN(
        input_size=m_max,
        hidden_size=student_model_cfg.get("hidden_size", 256),
        beta=student_model_cfg.get("beta", 0.9),
        threshold=student_model_cfg.get("threshold", 1.0),
        output_size=m_max,
        gradient_slope=student_model_cfg.get("gradient_slope", 25.0),
        learn_beta=student_model_cfg.get("learn_beta", True),
        num_layers=student_model_cfg.get("num_layers", 2),
        neuron_type=student_model_cfg.get("neuron_type", "rsynaptic"),
        alpha=student_model_cfg.get("alpha", 0.85),
    )
    student.to(device)

    student_params = sum(p.numel() for p in student.parameters())
    logger.info(
        "Student created: %d params (teacher: %d, ratio: %.2fx)",
        student_params, teacher_params, student_params / max(teacher_params, 1),
    )

    # ------------------------------------------------------------------
    # 9. Create distillation loss
    # ------------------------------------------------------------------
    distill_cfg = student_config.get("distillation", {})
    loss_cfg = student_config.get("loss", {})

    criterion = DistillationLoss(
        distill_weight=distill_cfg.get("distill_weight", 0.5),
        distill_weight_min=distill_cfg.get("distill_weight_min", None),
        distill_schedule=distill_cfg.get("distill_schedule", None),
        reg_weight=distill_cfg.get("reg_weight", 0.001),
        reg_type=distill_cfg.get("reg_type", "l1"),
        log_input=loss_cfg.get("log_input", False),
    )

    # ------------------------------------------------------------------
    # 10. Create experiment folder
    # ------------------------------------------------------------------
    combined_config = {
        "teacher": teacher_config,
        "student": student_config,
        "data": data_config,
    }
    exp_dir = create_experiment(
        slug=args.slug,
        config=combined_config,
        command=" ".join(sys.argv),
        notes=f"SNN distillation: teacher={args.teacher_s3_slug or args.teacher_checkpoint}, "
              f"student_hidden={student_model_cfg.get('hidden_size', 256)}, "
              f"distill_weight={distill_cfg.get('distill_weight', 0.5)}",
    )
    logger.info("Experiment folder: %s", exp_dir)

    # ------------------------------------------------------------------
    # 11. Upload metadata to S3 before training (crash safety)
    # ------------------------------------------------------------------
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        try:
            from scripts.nrp_train import upload_experiment_metadata_to_s3
            upload_experiment_metadata_to_s3(exp_dir)
        except ImportError:
            logger.warning("Could not upload metadata to S3")

    # ------------------------------------------------------------------
    # 12. Initialize W&B if available
    # ------------------------------------------------------------------
    wandb_run = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            wandb_run = wandb.init(
                project="spike-prophecy",
                name=args.slug,
                config=combined_config,
                tags=["phase4", "distillation", "snn"],
            )
            logger.info("W&B initialized: %s", wandb_run.url)
        except ImportError:
            logger.warning("wandb not installed — skipping")

    # ------------------------------------------------------------------
    # 13. Train with DistillTrainer
    # ------------------------------------------------------------------
    logger.info("Starting SNN distillation training...")

    # Wire S3 callbacks for crash-safe uploads
    checkpoint_callback = None
    metrics_callback = None
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        try:
            from scripts.nrp_train import (
                upload_checkpoint_to_s3,
                upload_metrics_to_s3,
            )
            checkpoint_callback = upload_checkpoint_to_s3
            metrics_callback = lambda exp=exp_dir: upload_metrics_to_s3(exp, 0)
        except ImportError:
            pass

    trainer = DistillTrainer(
        model=student,
        train_loader=distill_loaders["train"],
        val_loader=distill_loaders["val"],
        config=student_config,
        device=device,
        criterion=criterion,
        exp_dir=exp_dir,
    )

    # Wire callbacks if available
    if checkpoint_callback:
        trainer.checkpoint_callback = checkpoint_callback
    if metrics_callback:
        trainer.metrics_callback = metrics_callback

    history = trainer.train()

    # ------------------------------------------------------------------
    # 14. Log final metrics to W&B
    # ------------------------------------------------------------------
    if wandb_run is not None:
        try:
            # Log final validation metrics
            if history and len(history) > 0:
                last = history[-1] if isinstance(history, list) else history
                for key, value in last.items():
                    if isinstance(value, (int, float)):
                        wandb_run.summary[key] = value
            wandb_run.finish()
        except Exception as e:
            logger.warning("W&B logging failed: %s", e)

    # ------------------------------------------------------------------
    # 15. Upload full experiment to S3
    # ------------------------------------------------------------------
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        logger.info("Uploading experiment to S3...")
        try:
            from scripts.nrp_train import upload_experiment_to_s3
            upload_experiment_to_s3(exp_dir)
        except ImportError:
            logger.warning("Could not import S3 upload — skipping")

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("SNN DISTILLATION COMPLETE — %.1f minutes", elapsed / 60)
    logger.info(
        "Student: %d params | Teacher: %d params | Ratio: %.2fx",
        student_params, teacher_params, student_params / max(teacher_params, 1),
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
