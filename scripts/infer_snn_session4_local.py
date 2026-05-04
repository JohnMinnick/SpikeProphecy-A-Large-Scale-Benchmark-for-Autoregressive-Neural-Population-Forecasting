"""Local SNN inference on Steinmetz session 4 (no mamba-ssm needed).

Mirrors the NRP inference pipeline for SNN only:
  - Transpose npy from (n_neurons, n_bins) -> (T_total, n_neurons)
  - Zero-pad to M_max=1240
  - Use shared input/output projections (session_dims=None)
  - Trim outputs to actual n_neurons + first 660 timesteps
  - SNN model returns (rate, spikes) tuple — unwrap to rate

Output: data/figure_cache/_inference_workdir/snn_session4_local.npz
"""

import os
import json
import sys
import time
from pathlib import Path

import boto3
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.student import StudentSNN

LOCAL_CACHE = ROOT / "data" / "figure_cache" / "_inference_workdir"
OUT = LOCAL_CACHE / "snn_session4_local.npz"
SESSION_IDX = 4
HISTORY_BINS = 10
M_TRAIN = 1240
DISPLAY_LEN = 660

s3 = boto3.client(
    "s3", endpoint_url="https://s3-west.nrp-nautilus.io",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)


def fetch(key, dest):
    if dest.exists():
        return dest
    print(f"  fetching {key.split('/')[-1]}")
    s3.download_file("braingeneersdev", key, str(dest))
    return dest


def window(data, T):
    n = data.shape[0] - T
    X = np.zeros((n, T, data.shape[1]), dtype=np.float32)
    y = np.zeros((n, data.shape[1]), dtype=np.float32)
    for i in range(n):
        X[i] = data[i:i + T]
        y[i] = data[i + T]
    return X, y


def _unwrap(pred):
    if isinstance(pred, dict):
        for k in ("rate", "rates", "output", "out"):
            if k in pred:
                return pred[k]
        return list(pred.values())[0]
    if isinstance(pred, (list, tuple)):
        return pred[0]
    return pred


def main():
    LOCAL_CACHE.mkdir(parents=True, exist_ok=True)

    # Load session metadata + npy
    fetch("<anon>/spike-prophecy/inputs/steinmetz-session-cache/metadata.json",
          LOCAL_CACHE / "metadata.json")
    md = json.load(open(LOCAL_CACHE / "metadata.json"))
    sess = md["sessions"][SESSION_IDX]
    n_actual = sess.get("num_units")
    n_bins = sess.get("num_bins")
    npy_name = sess.get("npy", f"session_{SESSION_IDX:03d}.npy")
    fetch(f"<anon>/spike-prophecy/inputs/steinmetz-session-cache/{npy_name}",
          LOCAL_CACHE / npy_name)
    raw = np.load(str(LOCAL_CACHE / npy_name)).astype(np.float32)
    if raw.shape[0] == n_actual and raw.shape[1] == n_bins:
        raw = raw.T
    print(f"raw transposed: {raw.shape}; n_actual={n_actual}")

    # Zero-pad to M_TRAIN
    raw_padded = np.zeros((raw.shape[0], M_TRAIN), dtype=np.float32)
    raw_padded[:, :n_actual] = raw[:, :n_actual]
    X, y = window(raw_padded, HISTORY_BINS)
    print(f"windowed: X={X.shape}")
    y_actual = y[:, :n_actual]

    # SNN checkpoint + config
    snn_yaml = ROOT / "configs" / "student" / "standalone_snn_3l.yaml"
    snn_ckpt_key = ("<anon>/spike-prophecy/outputs/"
                    "2026-04-22_snn-standalone-3l-steinmetz/best_model.pt")
    fetch(snn_ckpt_key, LOCAL_CACHE / "ckpt_snn.pt")
    config = yaml.safe_load(open(snn_yaml))

    print("\n--- SNN (3L standalone, local) ---")
    ckpt = torch.load(str(LOCAL_CACHE / "ckpt_snn.pt"),
                      map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    model = StudentSNN.from_config(config, M_TRAIN)
    miss, unexp = model.load_state_dict(state, strict=False)
    if miss:
        print(f"  WARN missing: {len(miss)} (e.g. {miss[:2]})")
    if unexp:
        print(f"  WARN unexpected: {len(unexp)} (e.g. {unexp[:2]})")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    out = np.zeros((X.shape[0], n_actual), dtype=np.float32)
    batch = 64
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device)
            try:
                pred = model(xb)
            except TypeError:
                mask = torch.ones(xb.shape[0], xb.shape[2], device=device)
                pred = model(xb, mask=mask)
            pred = _unwrap(pred)
            arr = pred.detach().float().cpu().numpy()
            out[i:i + batch] = arr[:, :n_actual]
    dt = time.time() - t0
    print(f"  SNN: shape={out.shape}, {dt:.1f}s")

    # Sanity: pop_r vs gt
    from scipy.stats import pearsonr
    r = pearsonr(y_actual.sum(axis=1), out.sum(axis=1))[0]
    print(f"  SNN pop_r vs gt: {r:.4f}")

    # Trim to display length
    out_disp = out[:DISPLAY_LEN]
    gt_disp = y_actual[:DISPLAY_LEN]

    np.savez_compressed(
        str(OUT),
        snn_rates=out_disp.astype(np.float32),
        gt=gt_disp.astype(np.float32),
        session_idx=np.int64(SESSION_IDX),
        m_actual=np.int64(n_actual),
    )
    print(f"\nSaved: {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
