"""Standalone Mamba full-session prediction saver for NRP.

Downloads checkpoint + Steinmetz cache, runs inference with split=full,
saves per-session predictions to S3.
"""

import json
import os
import sys
import time
from pathlib import Path

import boto3
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/workspace")

from src.models.common import create_teacher_model
from src.utils.config import load_config
from src.utils.device import resolve_device
from src.data.multi_session_loader import (
    MaskedSpikeCountDataset,
    pad_to_channels,
    build_channel_mask,
)


B = "braingeneersdev"
CHECKPOINT_SLUG = "2026-03-26_baseline-mamba-v12"
CACHE_S3 = "jrm/spike-prophecy/inputs/steinmetz-session-cache"
OUTPUT_PREFIX = (
    "jrm/spike-prophecy/outputs/behavioral-predictions-steinmetz/mamba"
)


def main():
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ.get("ENDPOINT"),
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    device = resolve_device()
    print(f"Device: {device}")

    # Download checkpoint + config
    os.makedirs("/tmp/ckpt", exist_ok=True)
    for f in ["best_model.pt", "config.yaml"]:
        key = f"jrm/spike-prophecy/outputs/{CHECKPOINT_SLUG}/{f}"
        local = f"/tmp/ckpt/{f}"
        s3.download_file(B, key, local)
        print(f"  downloaded {f}")

    # Download Steinmetz cache
    os.makedirs("/tmp/cache", exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    n = 0
    for page in paginator.paginate(Bucket=B, Prefix=CACHE_S3 + "/"):
        for o in page.get("Contents", []):
            fn = o["Key"].split("/")[-1]
            if fn.endswith(".npy") or fn.endswith(".json"):
                local = f"/tmp/cache/{fn}"
                if not os.path.exists(local):
                    s3.download_file(B, o["Key"], local)
                    n += 1
    print(f"  cache: {n} files downloaded")

    # Load metadata
    meta = json.load(open("/tmp/cache/metadata.json"))
    m_max = meta["m_max"]
    history_bins = meta.get("history_bins", 10)
    n_sess = meta["num_sessions"]
    print(
        f"  m_max={m_max} sessions={n_sess} history={history_bins}"
    )

    # Build model
    cfg = load_config("/tmp/ckpt/config.yaml")
    cfg_block = cfg.get("teacher", cfg)
    model = create_teacher_model(
        config=cfg_block, input_size=m_max, session_dims=None
    )
    ckpt = torch.load(
        "/tmp/ckpt/best_model.pt", map_location=device, weights_only=False
    )
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    print(
        f"  loaded: {sum(p.numel() for p in model.parameters()):,} params"
    )

    # Inference per session
    out_dir = "/tmp/preds_out"
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    for sess_idx in range(n_sess):
        sess_info = meta["sessions"][sess_idx]
        m_i = sess_info["num_units"]
        npy_path = f"/tmp/cache/session_{sess_idx:03d}.npy"
        if not os.path.exists(npy_path):
            continue
        counts = np.load(npy_path).astype(np.int32)
        T_full = counts.shape[1]
        if T_full <= history_bins:
            continue

        padded = pad_to_channels(counts, m_max)
        mask_index = np.zeros(padded.shape[1], dtype=np.int32)
        ds = MaskedSpikeCountDataset(
            spike_counts=padded,
            mask_index=mask_index,
            session_masks=build_channel_mask(m_i, m_max).reshape(1, -1),
            history_bins=history_bins,
            output_channels=m_max,
        )
        if len(ds) == 0:
            continue
        dl = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

        all_pred, all_gt = [], []
        with torch.no_grad():
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
                all_pred.append(
                    rates[:, :m_i].cpu().numpy().astype(np.float32)
                )
                all_gt.append(
                    y[:, :m_i].cpu().numpy().astype(np.float32)
                )

        preds = np.concatenate(all_pred, axis=0).T  # (M_i, T)
        gts = np.concatenate(all_gt, axis=0).T
        local_out = f"{out_dir}/session_{sess_idx:03d}.npz"
        np.savez_compressed(
            local_out,
            pred_rates=preds,
            gt=gts,
            m_actual=m_i,
            session_idx=sess_idx,
            split_start_bin=history_bins,
        )

        # Upload immediately
        s3_key = f"{OUTPUT_PREFIX}/session_{sess_idx:03d}.npz"
        s3.upload_file(local_out, B, s3_key)
        elapsed = time.time() - t0
        print(
            f"  s{sess_idx:03d}: m={m_i} T={preds.shape[1]} "
            f"uploaded {s3_key} | {elapsed:.0f}s"
        )

    print("DONE")


if __name__ == "__main__":
    main()
