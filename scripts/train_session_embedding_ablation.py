"""Per-session input embedding ablation (NeurIPS reviewer 2 ask).

Self-contained training script that adds a per-session learned bias vector
to the input projection of a Mamba teacher. Tests whether session-specific
adaptation closes the per-neuron Pearson r gap that the reviewer flagged
as a likely consequence of the shared M_max x d input projection.

Architecture change vs nrp_teacher_mamba.yaml baseline:
  + nn.Embedding(num_sessions, hidden_size)
  + In forward: projected = input_norm(input_proj(x) + session_emb(s_idx))
Everything else identical (3-layer Mamba, hidden=256, etc).

Saves:
  outputs/session_embedding_ablation/best_model.pt
  outputs/session_embedding_ablation/metrics.json

Usage:
    python scripts/train_session_embedding_ablation.py \
        --data-config configs/data/steinmetz_multi_nrp_50ms_no_cov.yaml \
        --teacher-config configs/teacher/nrp_teacher_mamba.yaml \
        --slug session-emb-mamba-s42 \
        --epochs 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.device import resolve_device
from src.utils.seed import seed_everything
from src.data.multi_session_loader import (
    MaskedSpikeCountDataset, build_channel_mask, preprocess_and_cache,
)
from src.models.mamba_baseline import TeacherMamba

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("session_emb_ablation")


class SessionAwareDataset(torch.utils.data.Dataset):
    """Wraps a MaskedSpikeCountDataset and additionally returns session_idx."""

    def __init__(self, base):
        self.base = base
        # Cache mask_index for fast lookup
        self._mask_index = base._mask_index
        self._history_bins = base.history_bins
        self._valid_indices = base._valid_indices

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        out = self.base[idx]
        # The 'mask_index' for this sample's TARGET bin gives session_idx
        t = self._valid_indices[idx]
        session_idx = int(self._mask_index[t + self._history_bins])
        # Append session_idx as the last element. We return:
        # (x, y, mask, session_idx) for the no-covariates case.
        if len(out) == 3:
            x, y, mask = out
            return x, y, mask, session_idx
        elif len(out) == 4:
            x, y, mask, cov = out
            return x, y, mask, cov, session_idx
        else:
            raise ValueError(f"unexpected base output len: {len(out)}")


class MambaWithSessionEmbedding(nn.Module):
    """Wraps a TeacherMamba and adds a per-session input bias."""

    def __init__(self, mamba: TeacherMamba, num_sessions: int):
        super().__init__()
        self.mamba = mamba
        self.session_emb = nn.Embedding(num_sessions, mamba.hidden_size)
        nn.init.zeros_(self.session_emb.weight)  # start as identity

    def forward(self, x, session_idx=None, **kwargs):
        # Replicate TeacherMamba.forward() but inject session bias after
        # the input projection.
        m = self.mamba
        # Input projection
        projected = m.input_proj(x)  # (B, T, d_model)
        if session_idx is not None:
            # session_idx: (B,) long
            bias = self.session_emb(session_idx)  # (B, d_model)
            projected = projected + bias.unsqueeze(1)  # broadcast over T
        projected = m.input_norm(projected)
        hidden = projected
        for block in m.mamba_blocks:
            hidden = block(hidden)
        hidden = m.final_norm(hidden)
        if m.attn_query is not None:
            attn_scores = m.attn_query(hidden)
            attn_weights = torch.softmax(attn_scores, dim=1)
            context = (hidden * attn_weights).sum(dim=1)
        else:
            context = hidden[:, -1, :]
        context = m.output_norm(context)
        raw = m.output_proj(context)
        rates = m.softplus(raw)
        return rates


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-config", required=True)
    p.add_argument("--teacher-config", required=True)
    p.add_argument("--slug", default="session-emb-mamba")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="outputs/session_embedding_ablation")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-train-batches", type=int, default=None,
                   help="Cap batches per epoch (smoke test).")
    return p.parse_args()


def per_neuron_r(pred, gt, mask):
    """Mean per-neuron Pearson r across real channels."""
    # pred, gt: (N_samples, M_max)   mask: (N_samples, M_max) {0,1}
    rs = []
    for ch in range(pred.shape[1]):
        m = mask[:, ch] > 0.5
        if m.sum() < 20:
            continue
        p = pred[m, ch]
        g = gt[m, ch]
        sp, sg = p.std(), g.std()
        if sp < 1e-9 or sg < 1e-9:
            continue
        rs.append(float(((p - p.mean()) * (g - g.mean())).mean() / (sp * sg)))
    return float(np.mean(rs)) if rs else 0.0


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    log.info("device=%s seed=%d", device, args.seed)

    teacher_cfg = load_config(args.teacher_config)
    data_cfg = load_config(args.data_config)
    teacher_cfg["seed"] = args.seed

    # On NRP the NWB files aren't shipped with the image — download from S3.
    if not Path("data/raw").exists() or not list(Path("data/raw").glob("*.nwb")):
        try:
            sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
            from nrp_train import download_nwb_from_s3
            log.info("NRP mode — downloading NWBs from S3 ...")
            download_nwb_from_s3()
        except Exception as e:
            log.warning("NWB download skipped: %s", e)

    log.info("Preprocessing data ...")
    cache_dir, multi_meta = preprocess_and_cache(data_cfg)
    log.info(
        "  m_max=%d, num_sessions=%d",
        multi_meta["m_max"], multi_meta["num_sessions"],
    )

    # Load cached spike counts directly from .npy files. Each session is
    # stored as session_NNN.npy (M_actual, T) — pad to M_max and concat.
    cache_dir = Path(cache_dir)
    M_max = multi_meta["m_max"]
    num_sessions = multi_meta["num_sessions"]

    session_arrays = []
    session_masks = np.zeros((num_sessions, M_max), dtype=np.float32)
    mask_index_pieces = []
    for s in multi_meta["sessions"]:
        sidx = s["index"]
        n_units = s["num_units"]
        npy = cache_dir / f"session_{sidx:03d}.npy"
        counts = np.load(npy)  # (M_actual, T)
        # Pad to M_max
        padded = np.zeros((M_max, counts.shape[1]), dtype=counts.dtype)
        padded[:n_units] = counts
        session_arrays.append(padded)
        session_masks[sidx, :n_units] = 1.0
        mask_index_pieces.append(np.full(counts.shape[1], sidx, dtype=np.int32))
    spike_counts = np.concatenate(session_arrays, axis=1)
    mask_index = np.concatenate(mask_index_pieces, axis=0)
    log.info("Concatenated spike_counts shape: %s", spike_counts.shape)

    # Build train/val splits
    history_bins = 10
    base_train = MaskedSpikeCountDataset(
        spike_counts=spike_counts,
        mask_index=mask_index,
        session_masks=session_masks,
        history_bins=history_bins,
    )
    # 80/20 split via valid indices
    n_total = len(base_train)
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n_total)
    n_val = int(n_total * 0.2)
    val_idx = set(perm[:n_val].tolist())

    class IdxFilteredDataset(torch.utils.data.Dataset):
        def __init__(self, base, keep_idx_set):
            self.base = SessionAwareDataset(base)
            self.idx = sorted(keep_idx_set)
        def __len__(self):
            return len(self.idx)
        def __getitem__(self, i):
            return self.base[self.idx[i]]

    train_ds = IdxFilteredDataset(base_train, set(range(n_total)) - val_idx)
    val_ds = IdxFilteredDataset(base_train, val_idx)
    log.info("split: train=%d  val=%d", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # Build model
    model_cfg = teacher_cfg["model"]
    base_mamba = TeacherMamba(
        input_size=M_max,
        hidden_size=model_cfg.get("hidden_size", 256),
        num_layers=model_cfg.get("num_layers", 3),
        d_state=model_cfg.get("d_state", 16),
        d_conv=model_cfg.get("d_conv", 4),
        expand=model_cfg.get("expand", 2),
        dropout=model_cfg.get("dropout", 0.2),
        use_layer_norm=model_cfg.get("use_layer_norm", True),
        use_attention=model_cfg.get("use_attention", False),
    )
    model = MambaWithSessionEmbedding(base_mamba, num_sessions).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("model params: %d (incl. %d session-emb params)",
             n_params, sum(p.numel() for p in model.session_emb.parameters()))

    # Optimizer
    train_cfg = teacher_cfg.get("training", {})
    lr = args.lr if args.lr else train_cfg.get("learning_rate", 5e-4)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    # Train
    out_dir = Path(args.output_dir) / args.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_r = -1.0
    history = []

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        train_loss = 0.0; n_batches = 0
        for bi, batch in enumerate(train_loader):
            if args.max_train_batches and bi >= args.max_train_batches:
                break
            x, y, mask, sidx = batch
            x = x.to(device); y = y.to(device); mask = mask.to(device)
            sidx = sidx.to(device).long()
            rates = model(x, session_idx=sidx)
            # Poisson NLL (only on real channels)
            eps = 1e-7
            nll_per = (rates - y * torch.log(rates + eps)) * mask
            loss = nll_per.sum() / (mask.sum() + 1e-7)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item(); n_batches += 1
        train_loss /= max(n_batches, 1)

        # Validate
        model.eval()
        all_pred, all_gt, all_mask = [], [], []
        val_loss = 0.0; n_vb = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y, mask, sidx = batch
                x = x.to(device); y = y.to(device); mask = mask.to(device)
                sidx = sidx.to(device).long()
                rates = model(x, session_idx=sidx)
                eps = 1e-7
                nll = ((rates - y * torch.log(rates + eps)) * mask).sum() / (mask.sum() + 1e-7)
                val_loss += nll.item(); n_vb += 1
                all_pred.append(rates.cpu().numpy())
                all_gt.append(y.cpu().numpy())
                all_mask.append(mask.cpu().numpy())
        val_loss /= max(n_vb, 1)

        pred = np.concatenate(all_pred); gt = np.concatenate(all_gt); msk = np.concatenate(all_mask)
        val_r = per_neuron_r(pred, gt, msk)
        elapsed = time.time() - t0
        log.info(
            "epoch %d | train=%.4f val=%.4f val_r=%.4f | %.1fs",
            epoch, train_loss, val_loss, val_r, elapsed,
        )
        history.append({
            "epoch": epoch, "train_nll": train_loss,
            "val_nll": val_loss, "val_per_neuron_r": val_r,
            "elapsed_s": elapsed,
        })

        if val_r > best_val_r:
            best_val_r = val_r
            torch.save({
                "model_state": model.state_dict(),
                "val_r": val_r,
                "epoch": epoch,
                "n_params": n_params,
                "num_sessions": num_sessions,
                "M_max": M_max,
            }, out_dir / "best_model.pt")
            log.info("  -> new best, saved")

    # Save metrics
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "slug": args.slug,
            "seed": args.seed,
            "best_val_per_neuron_r": best_val_r,
            "n_params": n_params,
            "num_sessions": num_sessions,
            "history": history,
        }, f, indent=2)
    log.info("DONE. Best val per-neuron r = %.4f", best_val_r)
    log.info("Saved: %s", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
