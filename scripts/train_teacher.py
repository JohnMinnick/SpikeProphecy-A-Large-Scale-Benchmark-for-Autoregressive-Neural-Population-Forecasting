"""
Train the teacher ANN on synthetic spike-count data.

Thin entrypoint script that:
    1. Loads configs (data + teacher)
    2. Generates/loads synthetic data
    3. Bins spike trains → spike counts
    4. Creates DataLoaders via temporal split
    5. Trains the teacher LSTM with early stopping
    6. Evaluates against baselines
    7. Saves experiment (config, metrics, plots, checkpoints)

Usage:
    python scripts/train_teacher.py
    python scripts/train_teacher.py --data-config configs/data/quickrun.yaml
    python scripts/train_teacher.py --epochs 50 --lr 0.0005
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path for `src.*` imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.spikeinterface_generator import generate_synthetic_recording
from src.data.modulated_generator import generate_modulated_spikes
from src.data.real_data_loader import load_nwb_spikes
from src.data.multi_session_loader import (
    load_multi_session_nwb,
    create_masked_dataloaders,
    preprocess_and_cache,
    create_dataloaders as create_multi_dataloaders,
)
from src.data.binning import bin_spike_trains
from src.data.spike_dataset import create_dataloaders
from src.eval.metrics import compute_all_baselines
from src.models.teacher import TeacherLSTM, create_teacher_model
from src.train.trainer import Trainer
from src.utils.config import load_config
from src.utils.device import resolve_device, log_device_info
from src.utils.experiment import create_experiment, save_metrics
from src.viz.training_plots import (
    plot_loss_curves,
    plot_lr_schedule,
    plot_metric_curves,
    plot_prediction_vs_actual,
    plot_split_comparison,
)
from src.viz.style import save_figure

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Set random seed: %d", seed)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the teacher ANN on synthetic spike-count data."
    )
    parser.add_argument(
        "--teacher-config",
        type=str,
        default="configs/teacher/default.yaml",
        help="Path to teacher config YAML.",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="configs/data/default.yaml",
        help="Path to data config YAML.",
    )
    parser.add_argument(
        "--slug",
        type=str,
        default="teacher_baseline",
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
        "--extract-targets",
        action="store_true",
        help="Run distillation target extraction after training.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint (.pt) to resume training from.",
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
        help="Override the seed in teacher_config['seed']. Used for "
             "multi-seed sweeps (NeurIPS 2026 reviewer ask).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entrypoint: generate data → train → evaluate → save."""
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load configs
    # ------------------------------------------------------------------
    logger.info("Loading configs...")
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

    # Merge seed from teacher config into data config if needed
    seed = teacher_config.get("seed", 42)
    set_seed(seed)

    # ------------------------------------------------------------------
    # 2. Resolve device
    # ------------------------------------------------------------------
    compute_cfg = teacher_config.get("compute", {})
    device = resolve_device(compute_cfg.get("device", "auto"))
    log_device_info(device)

    # ------------------------------------------------------------------
    # 3. Generate/load data (dispatch on source.type)
    # ------------------------------------------------------------------
    source_type = data_config.get("source", {}).get("type", "spikeinterface")
    logger.info("Data source type: %s", source_type)

    # Flag for multi-session masked training
    is_multi_session = False
    multi_meta = None
    session_dims = None  # Session-specific heads (Phase 1, ADR-0013)

    if source_type == "nwb_multi":
        # Multi-session NWB: lazy per-session loading
        # Pass 1: preprocess and cache all sessions as .npy files
        logger.info("Preprocessing multi-session NWB data (caching)...")
        cache_dir, multi_meta = preprocess_and_cache(data_config)
        is_multi_session = True

        # Extract behavioral trial data BEFORE NWB cleanup.
        # This must happen here because NWB files are deleted below.
        # behavior_data is cached in memory (tiny — a few 1D arrays)
        # and attached to the trainer later for PSTH R² computation.
        _behavior_data = None
        try:
            from src.data.behavior_loader import (
                extract_trial_stimuli, compute_bin_edges,
            )
            source_glob = data_config.get(
                "source", {},
            ).get("glob", "data/raw/*.nwb")
            nwb_dir = Path(source_glob).parent
            nwb_files_for_behavior = sorted(nwb_dir.glob("*.nwb"))
            if nwb_files_for_behavior:
                nwb_path = str(nwb_files_for_behavior[0])
                # Get T from first session's cached data
                first_sess = multi_meta["sessions"][0]
                si = first_sess["index"]
                cache_path = cache_dir / f"session_{si:03d}.npy"
                if cache_path.exists():
                    first_counts = np.load(cache_path)
                    T_bins = first_counts.shape[1]  # (n_units, T)
                    bin_edges = compute_bin_edges(
                        T_bins, bin_width_ms=bin_width_ms,
                    )
                    _behavior_data = extract_trial_stimuli(
                        nwb_path, bin_edges,
                    )
                    n_trials = len(np.unique(
                        _behavior_data["trial_index"][
                            _behavior_data["trial_index"] >= 0
                        ],
                    ))
                    logger.info(
                        "Behavior data cached for PSTH R²: %d trials "
                        "from %s",
                        n_trials, nwb_files_for_behavior[0].name,
                    )
        except Exception as e:
            logger.warning(
                "Could not extract behavior data (PSTH R² skipped): %s", e,
            )

        # Free ephemeral storage by removing raw NWB files after caching.
        # The cached .npy files in cache_dir are all we need for training.
        # This prevents pod eviction on NRP (39 NWB files ≈ 20GB).
        source_glob = data_config.get("source", {}).get("glob", "data/raw/*.nwb")
        raw_dir = Path(source_glob).parent
        if raw_dir.is_dir():
            nwb_files = list(raw_dir.glob("*.nwb"))
            if nwb_files:
                total_mb = sum(f.stat().st_size for f in nwb_files) / (1024**2)
                for f in nwb_files:
                    f.unlink()
                logger.info(
                    "Cleaned up %d raw NWB files (%.0f MB freed)",
                    len(nwb_files), total_mb,
                )
        m_max = multi_meta["m_max"]
        num_sessions = multi_meta["num_sessions"]
        n_feat = multi_meta.get("n_features_per_channel", 0)
        logger.info(
            "Multi-session data: M_max=%d, %d sessions, %d history features/ch (lazy loading)",
            m_max, num_sessions, n_feat,
        )

        # For multi-session, input_size includes history features if cached
        # (EMA, ISI, etc. each add M_max channels to the input panel)
        input_size = m_max + n_feat * m_max if n_feat > 0 else m_max

        # Set model dimensions
        model_cfg = teacher_config.setdefault("model", {})
        model_cfg["input_size"] = input_size
        model_cfg["output_size"] = m_max

        # Wire covariates into model config (Tier 1 stimulus features, ADR-0012)
        n_cov = multi_meta.get("n_covariates", 0)
        cov_cfg = data_config.get("covariates", {})
        cov_mode = cov_cfg.get("mode", "additive")
        if n_cov > 0:
            model_cfg["n_covariates"] = n_cov
            model_cfg["covariate_mode"] = cov_mode
            cov_names = multi_meta.get("covariate_names", [])
            logger.info(
                "Covariates enabled: %d features, mode=%s, names=%s",
                n_cov, cov_mode, cov_names,
            )

        # Pass 2: create session-cycling DataLoaders (eager or lazy per config)
        loader_mode = data_config.get("loader_mode", "lazy")
        logger.info("Creating %s session-cycling DataLoaders...", loader_mode)
        loaders = create_multi_dataloaders(cache_dir, multi_meta, data_config)
        logger.info(
            "DataLoaders: train~%d batches, val~%d batches, test~%d batches",
            len(loaders["train"]), len(loaders["val"]), len(loaders["test"]),
        )

        # Use multi_meta for bin_metadata
        bin_metadata = {
            "source": "nwb_multi",
            "m_max": m_max,
            "num_sessions": num_sessions,
        }

        # Build session_dims for session-specific heads (opt-in via --session-heads)
        # Maps session_id -> N_i (real neuron count per session)
        # Default: shared heads (better generalization, val_r~0.48 vs ~0.14)
        if args.session_heads:
            session_dims = {
                f"session_{s['index']:03d}": s["num_units"]
                for s in multi_meta["sessions"]
            }
            logger.info(
                "Session-specific heads: %d sessions (min=%d, max=%d neurons)",
                len(session_dims),
                min(session_dims.values()),
                max(session_dims.values()),
            )
        else:
            logger.info("Using shared heads (default — best generalization)")

    elif source_type == "ibl":
        # IBL Brain-wide Map data: use pre-cached .npy files if available,
        # otherwise download via ONE API → bin → cache.
        # S3_IBL_CACHE_DIR env var allows combined runs to use a different dir.
        import os as _os
        ibl_cache_name = _os.environ.get(
            "S3_IBL_CACHE_DIR", "ibl_repeated_site_cache",
        )
        ibl_cache_dir = f"data/processed/{ibl_cache_name}"
        logger.info("Preprocessing IBL sessions (cache: %s)...", ibl_cache_dir)

        # If a valid metadata.json already exists (e.g., from a pre-built
        # combined cache downloaded from S3), skip re-processing entirely.
        import json as _json
        _meta_path = Path(ibl_cache_dir) / "metadata.json"
        if _meta_path.exists():
            with open(_meta_path, "r", encoding="utf-8") as _mf:
                multi_meta = _json.load(_mf)
            cache_dir = Path(ibl_cache_dir)
            logger.info(
                "Found existing cache metadata: %d sessions, M_max=%d — skipping re-processing",
                multi_meta["num_sessions"], multi_meta["m_max"],
            )
        else:
            from src.data.ibl_data_loader import preprocess_and_cache_ibl
            cache_dir, multi_meta = preprocess_and_cache_ibl(
                data_config, cache_dir=ibl_cache_dir,
            )
        is_multi_session = True

        m_max = multi_meta["m_max"]
        num_sessions = multi_meta["num_sessions"]
        n_feat = multi_meta.get("n_features_per_channel", 0)
        logger.info(
            "IBL data: M_max=%d, %d sessions (lazy loading)",
            m_max, num_sessions,
        )

        # Input/output dimensions (same logic as nwb_multi)
        input_size = m_max + n_feat * m_max if n_feat > 0 else m_max
        model_cfg = teacher_config.setdefault("model", {})
        model_cfg["input_size"] = input_size
        model_cfg["output_size"] = m_max

        # No covariates for IBL baseline
        n_cov = multi_meta.get("n_covariates", 0)
        if n_cov > 0:
            model_cfg["n_covariates"] = n_cov

        # Create session-cycling DataLoaders (same pipeline as NWB)
        loader_mode = data_config.get("loader_mode", "lazy")
        logger.info("Creating %s session-cycling DataLoaders...", loader_mode)
        loaders = create_multi_dataloaders(cache_dir, multi_meta, data_config)
        logger.info(
            "DataLoaders: train~%d batches, val~%d batches, test~%d batches",
            len(loaders["train"]), len(loaders["val"]), len(loaders["test"]),
        )

        bin_metadata = {
            "source": "ibl",
            "m_max": m_max,
            "num_sessions": num_sessions,
        }

        # Shared heads by default (same as NWB)
        if args.session_heads:
            session_dims = {
                f"session_{s['index']:03d}": s["num_units"]
                for s in multi_meta["sessions"]
            }
        else:
            logger.info("Using shared heads (default)")

    elif source_type == "modulated":
        # Modulated inhomogeneous Poisson generator (temporal structure)
        logger.info("Generating modulated spike trains...")
        sorting, _ = generate_modulated_spikes(data_config)
        recording = None  # No raw waveforms from modulated generator
    elif source_type == "nwb":
        # Real neural recordings from NWB files (e.g., Steinmetz 2019)
        logger.info("Loading real neural data from NWB file...")
        sorting, _ = load_nwb_spikes(data_config)
        recording = None  # NWB loader provides spike trains only
    else:
        # Default: SpikeInterface homogeneous Poisson + drifting recording
        logger.info("Generating synthetic recording (SpikeInterface)...")
        recording, sorting, extra_infos = generate_synthetic_recording(data_config)

    # ------------------------------------------------------------------
    # 4. Bin spike trains (single-session paths only)
    # ------------------------------------------------------------------
    # bin_width_ms is needed by both paths (also used in prediction plots)
    bin_width_ms = data_config.get("bin_width_ms", 10.0)
    if not is_multi_session:
        logger.info("Binning spike trains (Δt=%.1f ms)...", bin_width_ms)
        spike_counts, bin_metadata = bin_spike_trains(
            sorting, bin_width_ms=bin_width_ms, recording=recording,
        )
        m, t_total = spike_counts.shape
        logger.info("Spike counts: shape (%d, %d)", m, t_total)

        # Compute input_size including any history features (ADR-0009 Batch B)
        from src.data.history_features import compute_history_features
        hf_config = data_config.get("history_features", {})
        if hf_config.get("enabled", False):
            _, n_feat = compute_history_features(spike_counts, data_config)
            input_size = m + n_feat * m
            logger.info(
                "History features enabled: %d features × %d channels → "
                "input_size=%d", n_feat, m, input_size,
            )
        else:
            input_size = m
        # Set model dimensions: input may be expanded, output is always M channels
        model_cfg = teacher_config.setdefault("model", {})
        model_cfg["input_size"] = input_size
        model_cfg["output_size"] = m  # Targets are always original spike counts

        # ------------------------------------------------------------------
        # 5. Create DataLoaders (single-session)
        # ------------------------------------------------------------------
        logger.info("Creating DataLoaders...")
        loaders = create_dataloaders(spike_counts, data_config)
        logger.info(
            "DataLoaders: train=%d batches, val=%d batches, test=%d batches",
            len(loaders["train"]), len(loaders["val"]), len(loaders["test"]),
        )

    # ------------------------------------------------------------------
    # 6. Create experiment folder
    # ------------------------------------------------------------------
    combined_config = {
        "teacher": teacher_config,
        "data": data_config,
        "bin_metadata": bin_metadata,
    }
    exp_dir = create_experiment(
        slug=args.slug,
        config=combined_config,
        command=f"python scripts/train_teacher.py --teacher-config {args.teacher_config} --data-config {args.data_config}",
        notes="Initial teacher ANN baseline training on synthetic data.",
    )
    logger.info("Experiment folder: %s", exp_dir)

    # ------------------------------------------------------------------
    # 7. Train the model
    # ------------------------------------------------------------------
    arch = teacher_config.get("model", {}).get("architecture", "lstm")
    logger.info("Creating teacher model (architecture=%s)...", arch)
    model = create_teacher_model(
        teacher_config, input_size=input_size, session_dims=session_dims,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s, parameters: %d", type(model).__name__, n_params)

    # torch.compile() disabled — on NRP containers it returns an object
    # where .train() resolves to a function instead of nn.Module.train(),
    # crashing the Trainer.  The 15-30% speedup is not worth the risk.
    # Revisit if container PyTorch is updated or root cause is found.
    import platform
    if platform.system() != "Windows":
        logger.info("Skipping torch.compile() (disabled for NRP compatibility)")
    else:
        logger.info("Skipping torch.compile() on Windows (Triton unavailable)")

    # Build optional S3 checkpoint callback for NRP container runs.
    # When NRP_S3_CHECKPOINT=1 (set by nrp_train.py), the Trainer will
    # upload best_model.pt to S3 after every improvement — crash-safe.
    checkpoint_callback = None
    metrics_callback = None
    if os.environ.get("NRP_S3_CHECKPOINT") == "1":
        try:
            from scripts.nrp_train import upload_checkpoint_to_s3
            checkpoint_callback = upload_checkpoint_to_s3
            logger.info("S3 checkpoint callback enabled (NRP mode)")
        except ImportError:
            logger.warning(
                "NRP_S3_CHECKPOINT=1 but could not import "
                "upload_checkpoint_to_s3 — checkpoints will NOT be "
                "uploaded between epochs."
            )
        # Also build S3 metrics callback: uploads metrics.json after
        # each validation pass so partial results survive crashes.
        try:
            from scripts.nrp_train import upload_metrics_to_s3
            # Capture exp_dir in closure for the callback
            _exp_dir = exp_dir
            metrics_callback = lambda epoch, history: upload_metrics_to_s3(
                _exp_dir, epoch
            )
            logger.info("S3 metrics callback enabled (NRP mode)")
        except ImportError:
            logger.warning(
                "NRP_S3_CHECKPOINT=1 but could not import "
                "upload_metrics_to_s3 — metrics will NOT be "
                "uploaded incrementally."
            )

    trainer = Trainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        config=teacher_config,
        device=device,
        exp_dir=exp_dir,
        checkpoint_callback=checkpoint_callback,
        metrics_callback=metrics_callback,
    )

    # Build per-region Pearson r map for multi-session data.
    # Maps global channel indices to brain region names, enabling
    # evaluate() to log per-region Pearson r (e.g., val_pearson_r_VISp).
    if is_multi_session and multi_meta is not None:
        region_map = {}
        for sess in multi_meta.get("sessions", []):
            regions = sess.get("brain_regions")
            if regions is not None:
                # Each session's channels occupy [0, num_units) in M_max-padded space
                # Since per-session loading re-pads to M_max, we just need
                # to know which channels have which region within each session.
                # For the cycling loader, per-channel r is computed over M_max,
                # where channel i corresponds to unit i of whichever session is
                # being evaluated. The region mapping is session-global.
                for ch_idx, region in enumerate(regions):
                    if ch_idx not in region_map:
                        region_map[ch_idx] = region
        if region_map:
            trainer.region_map = region_map
            # Count unique regions for logging
            unique_regions = set(region_map.values())
            logger.info(
                "Region map: %d channels → %d unique regions",
                len(region_map), len(unique_regions),
            )

            # Wire up RegionHybridLoss if configured (KOSMOS Batch 3)
            loss_type = teacher_config.get("loss", {}).get("type", "poisson_nll")
            if loss_type == "region_hybrid":
                from src.train.region_loss import RegionHybridLoss
                m_max = multi_meta.get("m_max", model.output_size)
                region_loss = RegionHybridLoss(
                    region_map=region_map,
                    m_max=m_max,
                ).to(device)
                trainer.region_loss_fn = region_loss
                logger.info("Activated RegionHybridLoss (NegBin for hippo)")
            elif loss_type == "fano_adaptive":
                # FanoAdaptiveLoss is set up below after Fano factors
                pass  # handled in Fano factor block

        # Compute per-channel Fano factors from training data (KOSMOS).
        # Enables Fano-stratified Pearson r reporting (sub/near/super-Poisson).
        try:
            # preprocess_and_cache already imported at top of file
            cache_dir = Path(multi_meta.get("cache_dir", "data/processed"))
            m_max = multi_meta["m_max"]
            ff_accum = np.zeros(m_max, dtype=np.float64)
            ff_count = np.zeros(m_max, dtype=np.float64)
            for sess in multi_meta.get("sessions", []):
                si = sess["index"]
                npy_path = cache_dir / f"session_{si:03d}.npy"
                if npy_path.exists():
                    counts = np.load(npy_path).astype(np.float32)
                    n_units = counts.shape[0]
                    for j in range(n_units):
                        col = counts[j]
                        mu = col.mean()
                        if mu > 0.01:  # Skip near-silent neurons
                            ff_accum[j] += col.var() / mu
                            ff_count[j] += 1
            valid = ff_count > 0
            fano = np.ones(m_max, dtype=np.float32)  # Default FF=1
            fano[valid] = (ff_accum[valid] / ff_count[valid]).astype(
                np.float32,
            )
            trainer.fano_factors = fano
            n_sub = (fano[valid] < 1.0).sum()
            n_super = (fano[valid] > 1.5).sum()
            logger.info(
                "Fano factors: %d sub-Poisson, %d near-Poisson, "
                "%d super-Poisson (of %d active)",
                n_sub, int(valid.sum()) - n_sub - n_super,
                n_super, int(valid.sum()),
            )
        except Exception as e:
            logger.warning(
                "Could not compute Fano factors: %s", e,
            )

        # Attach behavior data for PSTH R² (cached before NWB cleanup)
        # _behavior_data is only set in the NWB code path; default to None
        # for IBL / pre-cached data sources that skip NWB loading.
        _behavior_data_local = locals().get("_behavior_data", None)
        if _behavior_data_local is not None:
            trainer.behavior_data = _behavior_data_local
            logger.info("Behavior data attached to trainer for PSTH R²")

    # Resume from checkpoint if requested
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            start_epoch = trainer.load_checkpoint(resume_path)
            logger.info("Resuming training from epoch %d", start_epoch)
        else:
            logger.warning("Resume checkpoint not found: %s", resume_path)

    # Eval-only mode: run evaluation with full multi-metric reporting
    if getattr(args, "eval_only", False):
        logger.info("=== EVAL-ONLY MODE ===")

        # Standard streaming eval
        val_metrics = trainer.evaluate(trainer.val_loader, prefix="val")
        logger.info("Standard evaluation results:")
        for k, v in sorted(val_metrics.items()):
            if isinstance(v, float):
                logger.info("  %s = %.4f", k, v)

        # Population metrics (co-BPS, calibration, PSTH R²)
        try:
            pop_metrics = trainer.evaluate_population(trainer.val_loader)
            pop_scalars = {
                k: v for k, v in pop_metrics.items()
                if isinstance(v, (int, float, type(None)))
            }
            logger.info("Population metrics:")
            for k, v in sorted(pop_scalars.items()):
                if isinstance(v, float):
                    logger.info("  %s = %.4f", k, v)
        except Exception as e:
            logger.warning("Population metrics failed: %s", e)
            pop_scalars = {}

        # Save all metrics
        eval_metrics = {
            "eval_only": True,
            **val_metrics,
            "population_metrics": pop_scalars,
        }
        save_metrics(exp_dir, eval_metrics, append=True)
        logger.info("Saved eval metrics to %s/metrics.json", exp_dir)
        return

    logger.info("Starting training...")
    history = trainer.train()

    # ------------------------------------------------------------------
    # 8. Save training metrics immediately (crash-safe)
    # ------------------------------------------------------------------
    teacher_metrics = {
        "teacher_best_val_loss": trainer.best_val_loss,
        "teacher_final_val_loss": history["val_loss"][-1],
        "teacher_best_val_pearson_r": max(history["val_pearson_r"]),
        "teacher_final_val_pearson_r": history["val_pearson_r"][-1],
        "teacher_final_val_mae": history["val_mae"][-1],
        "teacher_final_val_mse": history["val_mse"][-1],
        "teacher_n_params": n_params,
        "teacher_n_epochs_trained": len(history["train_loss"]),
        "history": history,
    }
    all_metrics = {**teacher_metrics}
    save_metrics(exp_dir, all_metrics, append=True)
    logger.info("Saved training metrics to %s/metrics.json", exp_dir)

    # ------------------------------------------------------------------
    # 9. Compute baselines (single-session only — requires spike_counts)
    # ------------------------------------------------------------------
    if not is_multi_session:
        logger.info("Computing baselines...")
        history_bins = data_config.get("dataset", {}).get("history_bins", 50)
        baselines = compute_all_baselines(spike_counts, history_bins=history_bins)
        all_metrics["baselines"] = baselines
        save_metrics(exp_dir, all_metrics, append=True)
        logger.info("Persistence baseline NLL: %.4f", baselines["persistence"]["poisson_nll"])
        logger.info("Mean-rate baseline NLL: %.4f", baselines["mean_rate"]["poisson_nll"])
    else:
        logger.info("Skipping baselines (nwb_multi mode — no single spike_counts).")

    # ------------------------------------------------------------------
    # 10. Sanity check: evaluate on all splits
    # ------------------------------------------------------------------
    logger.info("Running train/val/test split evaluation (sanity check)...")

    # Reload best model for evaluation
    best_model_path = exp_dir / "best_model.pt"
    if best_model_path.exists():
        logger.info("Loading best model from %s", best_model_path)
        best_ckpt = torch.load(
            best_model_path, map_location=device, weights_only=False,
        )
        # Strip _orig_mod. prefix for torch.compile() compat
        sd = {
            k.replace("_orig_mod.", ""): v
            for k, v in best_ckpt["model_state_dict"].items()
        }
        model.load_state_dict(sd)

    split_metrics = {}
    for split_name, loader in loaders.items():
        metrics = trainer.evaluate(loader, prefix=split_name)
        split_metrics[split_name] = metrics
        for k, v in metrics.items():
            logger.info("  %s = %.6f", k, v)

    # Save split metrics alongside other metrics
    all_metrics["split_eval"] = split_metrics
    save_metrics(exp_dir, all_metrics, append=True)

    # Quick sanity check log
    train_nll = split_metrics.get("train", {}).get(
        "train_poisson_nll", float("inf"),
    )
    val_nll = split_metrics.get("val", {}).get(
        "val_poisson_nll", float("inf"),
    )
    if train_nll < val_nll:
        logger.info(
            "[OK] Train NLL (%.4f) < Val NLL (%.4f) -> model is learning",
            train_nll, val_nll,
        )
    else:
        logger.warning(
            "[!!] Train NLL (%.4f) >= Val NLL (%.4f) -> may not be learning!",
            train_nll, val_nll,
        )

    # ------------------------------------------------------------------
    # 10. Extract distillation targets (Optional)
    # ------------------------------------------------------------------
    if args.extract_targets:
        logger.info("Extracting distillation targets...")
        from src.distill.extract_targets import (
            extract_teacher_targets,
            save_distillation_targets,
        )
        targets = extract_teacher_targets(model, loaders, device)
        save_distillation_targets(targets, exp_dir / "targets")

    # ------------------------------------------------------------------
    # 11. Generate plots
    # ------------------------------------------------------------------
    logger.info("Generating training plots...")
    plots_dir = exp_dir / "plots"

    # Loss curves
    fig = plot_loss_curves(history)
    save_figure(fig, "loss_curves", output_dir=plots_dir, close=True)

    # LR schedule
    fig = plot_lr_schedule(history)
    save_figure(fig, "lr_schedule", output_dir=plots_dir, close=True)

    # Metric curves (2x2 grid)
    fig = plot_metric_curves(history)
    save_figure(fig, "metric_curves", output_dir=plots_dir, close=True)

    # Split comparison bar chart (sanity check)
    fig = plot_split_comparison(split_metrics)
    save_figure(fig, "split_comparison", output_dir=plots_dir, close=True)

    # Prediction vs actual (on test set)
    # NOTE: skipped for nwb_multi — the full test set is too large to
    # accumulate predictions, and spike_counts is not available.
    if not is_multi_session:
        logger.info("Generating prediction plots on test set...")
        model.eval()
        model.to(device)
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in loaders["test"]:
                # Handle both (x, y) and (x, y, mask) batches
                x = batch[0].to(device)
                y = batch[1]
                y_hat = model(x)
                all_preds.append(y_hat.cpu().numpy())
                all_targets.append(y.numpy())

        if all_preds:
            preds_np = np.concatenate(all_preds, axis=0)
            targets_np = np.concatenate(all_targets, axis=0)
            fig = plot_prediction_vs_actual(
                preds_np, targets_np,
                bin_width_ms=bin_width_ms,
                title="Teacher Predictions vs Actual (Test Set)",
            )
            save_figure(fig, "prediction_vs_actual", output_dir=plots_dir, close=True)
    else:
        logger.info("Skipping prediction plot (nwb_multi — test set too large).")

    # ------------------------------------------------------------------
    # 12. Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("Experiment: %s", exp_dir)
    logger.info("Epochs trained: %d", len(history["train_loss"]))
    logger.info("Best val NLL: %.4f", trainer.best_val_loss)
    logger.info("Best val Pearson r: %.4f", max(history["val_pearson_r"]))
    logger.info("Train NLL: %.4f (sanity check)", train_nll)
    if not is_multi_session:
        logger.info("Beats persistence: %s",
                    trainer.best_val_loss < baselines["persistence"]["poisson_nll"])
        logger.info("Beats mean-rate: %s",
                    trainer.best_val_loss < baselines["mean_rate"]["poisson_nll"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
