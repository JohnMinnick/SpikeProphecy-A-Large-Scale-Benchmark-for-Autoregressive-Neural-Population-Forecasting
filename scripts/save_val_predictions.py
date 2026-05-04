"""Save per-session val-split rate predictions for one architecture.

Output: outputs/eval_local/behavioral_predictions/{tag}/session_NNN.npz
with keys: pred_rates (M_i, T), session_idx, m_actual.

Supports teacher models (transformer/lru/lstm) and student SNN.
Mirrors the eval_local_corrected.py / eval_local_teacher.py logic.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.device import resolve_device
from src.data.multi_session_loader import (
    MaskedSpikeCountDataset,
    pad_to_channels,
    build_channel_mask,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Model config YAML (teacher or student).",
    )
    p.add_argument(
        "--kind",
        type=str,
        choices=["teacher", "student"],
        required=True,
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "session_cache"),
    )
    p.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Output folder name under behavioral_predictions/.",
    )
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-sessions", type=int, default=None)
    return p.parse_args()


def build_model(kind, config, m_max):
    cfg_block = config.get("teacher", config)
    model_cfg = cfg_block.get("model", config.get("model", {}))
    if kind == "teacher":
        from src.models.common import create_teacher_model
        return create_teacher_model(
            config=cfg_block, input_size=m_max, session_dims=None
        )
    else:
        from src.models.student import StudentSNN
        return StudentSNN(
            input_size=m_max,
            output_size=m_max,
            hidden_size=model_cfg.get("hidden_size", 256),
            beta=model_cfg.get("beta", 0.9),
            threshold=model_cfg.get("threshold", 1.0),
            gradient_slope=model_cfg.get("gradient_slope", 25.0),
            learn_beta=model_cfg.get("learn_beta", True),
            num_layers=model_cfg.get("num_layers", 1),
            neuron_type=model_cfg.get("neuron_type", "rsynaptic"),
            alpha=model_cfg.get("alpha", 0.85),
            use_layer_norm=model_cfg.get("use_layer_norm", True),
            dropout=model_cfg.get("dropout", 0.0),
            learn_threshold=model_cfg.get("learn_threshold", False),
            readout_mode=model_cfg.get("readout_mode", "mean"),
            auxiliary_heads=model_cfg.get("auxiliary_heads", None),
        )


@torch.no_grad()
def main():
    args = parse_args()
    device = resolve_device()
    print(f"Device: {device}")

    cache_dir = Path(args.cache_dir)
    metadata = json.load(open(cache_dir / "metadata.json"))
    m_max = metadata["m_max"]
    history_bins = metadata.get("history_bins", 10)
    n_sessions = metadata["num_sessions"]
    print(
        f"Cache: {cache_dir} | M_max={m_max} | sessions={n_sessions} "
        f"| history={history_bins}"
    )

    cfg = load_config(args.config)
    model = build_model(args.kind, cfg, m_max)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        print(f"strict=True failed: {e}\nFalling back to strict=False")
        model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Loaded {args.ckpt}: kind={args.kind} params={n_params:,} "
        f"epoch={ckpt.get('epoch', '?')}"
    )

    out_dir = (
        PROJECT_ROOT / "outputs" / "eval_local"
        / "behavioral_predictions" / args.tag
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions_to_run = (
        n_sessions if args.max_sessions is None
        else min(args.max_sessions, n_sessions)
    )

    t0 = time.time()
    for sess_idx in range(sessions_to_run):
        sess_info = metadata["sessions"][sess_idx]
        m_i = sess_info["num_units"]
        npy_path = cache_dir / f"session_{sess_idx:03d}.npy"
        if not npy_path.exists():
            continue

        counts = np.load(npy_path)
        bounds = sess_info["split_boundaries"]
        if args.split == "val":
            split_start = bounds["train_end"]
            split_end = bounds["val_end"]
        elif args.split == "test":
            split_start = bounds["val_end"]
            split_end = sess_info["num_bins"]
        else:  # full, all bins
            split_start, split_end = 0, sess_info["num_bins"]

        split_counts = counts[:, split_start:split_end].astype(np.int32)
        del counts
        if split_counts.shape[1] <= history_bins:
            continue

        padded = pad_to_channels(split_counts, m_max)
        del split_counts
        mask_index = np.zeros(padded.shape[1], dtype=np.int32)
        ds = MaskedSpikeCountDataset(
            spike_counts=padded,
            mask_index=mask_index,
            session_masks=build_channel_mask(m_i, m_max).reshape(1, -1),
            history_bins=history_bins,
            output_channels=m_max,
        )
        del padded
        if len(ds) == 0:
            continue

        dl = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False, num_workers=0
        )

        all_pred, all_gt = [], []
        for batch in dl:
            x = batch[0].to(device)
            y = batch[1]
            out = model(x)
            if isinstance(out, dict):
                rates = out.get("rates", out.get("output"))
            elif isinstance(out, tuple):
                rates = out[0]
            else:
                rates = out
            all_pred.append(rates[:, :m_i].cpu().numpy().astype(np.float32))
            all_gt.append(y[:, :m_i].cpu().numpy().astype(np.float32))

        preds = np.concatenate(all_pred, axis=0).T  # (M_i, T)
        gts = np.concatenate(all_gt, axis=0).T
        out_path = out_dir / f"session_{sess_idx:03d}.npz"
        np.savez_compressed(
            out_path,
            pred_rates=preds,
            gt=gts,
            m_actual=m_i,
            session_idx=sess_idx,
            split_start_bin=int(split_start + history_bins),
        )
        elapsed = time.time() - t0
        print(
            f"  s{sess_idx:03d}: m_i={m_i} T={preds.shape[1]} "
            f"saved {out_path.name} | {elapsed:.0f}s"
        )

    print(f"Done. Wrote {args.tag}/ ({sessions_to_run} sessions).")


if __name__ == "__main__":
    main()
