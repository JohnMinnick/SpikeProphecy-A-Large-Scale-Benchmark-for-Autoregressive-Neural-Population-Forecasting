"""Local multi-architecture inference on Steinmetz session 4.

Runs each of 6 ANN architectures on the user's local GPU (RTX 4060)
and saves predictions to data/figure_cache/multi_arch_session4.npz.
SNN predictions are pulled from the existing cached NPZ.
"""

import os
import json
import time
from pathlib import Path

import boto3
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from src.models.common import create_teacher_model

OUT = ROOT / "data" / "figure_cache" / "multi_arch_session4.npz"
LOCAL_CACHE = ROOT / "data" / "figure_cache" / "_inference_workdir"
LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
SESSION_IDX = 4
HISTORY_BINS = 10

ARCHS = [
    # (arch_key, checkpoint S3 slug, config YAML local path, display name)
    ("mamba",       "2026-03-26_baseline-mamba-v12",
                    "configs/teacher/nrp_teacher_mamba.yaml",       "Mamba"),
    ("hgrn2",       "2026-04-21_2026-04-21_baseline-hgrn2-v1",
                    "configs/teacher/nrp_teacher_hgrn2.yaml",       "HGRN2"),
    ("transformer", "2026-03-25_baseline-transformer-v12",
                    "configs/archive/teacher/nrp_teacher_transformer.yaml", "Transformer"),
    ("gated_delta", "2026-04-22_2026-04-22_baseline-gated-delta-v1",
                    "configs/teacher/nrp_teacher_gated_delta.yaml", "GatedDeltaNet"),
    ("lru",         "2026-03-25_baseline-lru-v12",
                    "configs/teacher/nrp_teacher_lru_v2.yaml",      "LRU"),
    ("lstm",        "2026-04-15_baseline-lstm-v23",
                    "configs/archive/teacher/nrp_teacher.yaml",     "LSTM"),
]
SNN_CACHE_KEY = "jrm/spike-prophecy/outputs/full-inference-arrays/session_004.npz"
SESSION_CACHE_KEY_PREFIX = "jrm/spike-prophecy/inputs/steinmetz-session-cache/"

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


def load_session():
    metadata_path = fetch(
        SESSION_CACHE_KEY_PREFIX + "metadata.json",
        LOCAL_CACHE / "metadata.json",
    )
    metadata = json.load(open(metadata_path))
    sess_info = metadata["sessions"][SESSION_IDX]
    n_neurons = sess_info.get("num_units", sess_info.get("n_units"))
    n_bins = sess_info.get("num_bins")
    npy_name = sess_info.get("npy", f"session_{SESSION_IDX:03d}.npy")
    npy_path = fetch(SESSION_CACHE_KEY_PREFIX + npy_name,
                     LOCAL_CACHE / npy_name)
    raw = np.load(str(npy_path)).astype(np.float32)
    # The npy is stored as (n_neurons, n_bins). We want (T_total, M).
    if raw.shape[0] == n_neurons and raw.shape[1] == n_bins:
        raw = raw.T  # now (T_total, n_neurons)
    return raw, n_neurons, raw.shape[1]


def window(data, T):
    n = data.shape[0] - T
    X = np.zeros((n, T, data.shape[1]), dtype=np.float32)
    y = np.zeros((n, data.shape[1]), dtype=np.float32)
    for i in range(n):
        X[i] = data[i:i + T]
        y[i] = data[i + T]
    return X, y


def _call_model(model, xb, mask, session_id):
    """Try a sequence of forward signatures, return rate prediction."""
    attempts = [
        lambda: model(xb, mask=mask, session_id=session_id),
        lambda: model(xb, session_id=session_id),
        lambda: model(xb, channel_mask=mask, session_id=session_id),
        lambda: model(xb, mask=mask),
        lambda: model(xb),
    ]
    last_err = None
    for fn in attempts:
        try:
            return fn()
        except (TypeError, AssertionError) as e:
            last_err = e
            continue
    raise last_err


def run_inference(model, X, n_neurons_actual, session_id, batch=128,
                  device="cuda"):
    model.eval()
    out = np.zeros((X.shape[0], n_neurons_actual), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device)
            mask = torch.ones(xb.shape[0], xb.shape[2], device=device)
            pred = _call_model(model, xb, mask, session_id)
            if isinstance(pred, dict):
                pred = pred.get("rate", pred.get("output",
                                                 list(pred.values())[0]))
            arr = pred.detach().float().cpu().numpy()
            out[i:i + batch] = arr[:, :n_neurons_actual]
    return out


def main():
    print("=" * 60)
    print(" Local multi-arch inference: Steinmetz session 4")
    print("=" * 60)
    raw_full, n_actual, m_full_pad = load_session()
    print(f"  session_004 raw shape: {raw_full.shape}, actual neurons {n_actual}")

    # The checkpoints use SHARED input/output projections of width
    # M_max=1240.  Zero-pad to 1240 for input; trim output to n_actual.
    M_TRAIN = 1240
    raw = np.zeros((raw_full.shape[0], M_TRAIN), dtype=np.float32)
    raw[:, :n_actual] = raw_full[:, :n_actual]
    print(f"  zero-padded to M_max={M_TRAIN}: shape {raw.shape}")

    X, y = window(raw, HISTORY_BINS)
    print(f"  windowed: X={X.shape}, y={y.shape}")
    y_actual = y[:, :n_actual]

    out = {
        "gt": y_actual,
        "session_idx": np.int64(SESSION_IDX),
        "m_actual": np.int64(n_actual),
    }

    # Mamba uses mamba-ssm (Linux + CUDA only).  Fall back to cached
    # rates from data/figure_cache/session_004.npz if it exists.
    mamba_local_npz = ROOT / "data" / "figure_cache" / "session_004.npz"
    if mamba_local_npz.exists():
        mc = np.load(str(mamba_local_npz))
        if "mamba_rates" in mc.files:
            m_rates = mc["mamba_rates"]
            # Align time + trim to actual neurons
            if m_rates.shape[1] > n_actual:
                m_rates = m_rates[:, :n_actual]
            if m_rates.shape[0] != y_actual.shape[0]:
                n = min(m_rates.shape[0], y_actual.shape[0])
                m_rates = m_rates[:n]
            out["mamba_rates"] = m_rates
            print(f"  Mamba (from cache): shape={m_rates.shape}")

    for arch_key, slug, yaml_path, name in ARCHS:
        if arch_key == "mamba":
            continue  # handled above
        print(f"\n--- {name} ({arch_key}) ---")
        ckpt_dest = LOCAL_CACHE / f"ckpt_{arch_key}.pt"
        try:
            fetch(f"jrm/spike-prophecy/outputs/{slug}/best_model.pt", ckpt_dest)
        except Exception as e:
            print(f"  failed to fetch checkpoint: {e}")
            continue

        # Load YAML config
        ypath = ROOT / yaml_path
        if not ypath.exists():
            print(f"  YAML not found at {ypath}, skipping")
            continue
        config = yaml.safe_load(open(ypath))
        # Force the architecture key
        config.setdefault("model", {})["architecture"] = arch_key

        try:
            ckpt = torch.load(str(ckpt_dest), map_location="cpu",
                              weights_only=False)
            state = ckpt["model_state_dict"]
            # Use SHARED projections (no session_dims) since the
            # checkpoints contain shared input_proj/output_proj.
            model = create_teacher_model(config, M_TRAIN, session_dims=None)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"  WARN missing: {len(missing)} (e.g. {missing[:2]})")
            if unexpected:
                print(f"  WARN unexpected: {len(unexpected)} (e.g. {unexpected[:2]})")
            model = model.cuda()
            t0 = time.time()
            preds = run_inference(model, X, n_actual,
                                  session_id=f"session_{SESSION_IDX:03d}",
                                  device="cuda")
            print(f"  {name}: shape={preds.shape}, {time.time()-t0:.1f}s")
            out[f"{arch_key}_rates"] = preds
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {name}: FAILED ({type(e).__name__}: {e})")

    # SNN: pull from existing cache; align to actual neurons
    print("\n--- SNN (from cache) ---")
    snn_path = LOCAL_CACHE / "snn_cache.npz"
    fetch(SNN_CACHE_KEY, snn_path)
    snn_cache = np.load(str(snn_path))
    snn_rates = snn_cache["snn_rates"]
    if snn_rates.shape[1] > n_actual:
        snn_rates = snn_rates[:, :n_actual]
    if snn_rates.shape[0] != y_actual.shape[0]:
        n = min(snn_rates.shape[0], y_actual.shape[0])
        snn_rates = snn_rates[:n]
        # also trim gt to match
        out["gt"] = out["gt"][:n]
    out["snn_rates"] = snn_rates
    print(f"  SNN: shape={snn_rates.shape}")

    # Trim every prediction array to a common ~33s window (660 bins)
    # so the saved NPZ stays small enough for the figure cache.
    DISPLAY_LEN = 660
    common_t = min(
        DISPLAY_LEN,
        *[v.shape[0] for k, v in out.items()
          if hasattr(v, "shape") and v.ndim == 2]
    )
    for k in list(out.keys()):
        v = out[k]
        if hasattr(v, "shape") and v.ndim == 2 and v.shape[0] > common_t:
            out[k] = v[:common_t]
    print(f"\nTrimmed all arrays to first {common_t} bins "
          f"({common_t * 0.05:.1f}s)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(OUT), **out)
    print(f"Saved: {OUT} ({OUT.stat().st_size:,} bytes)")
    print("Keys:", list(out.keys()))


if __name__ == "__main__":
    main()
