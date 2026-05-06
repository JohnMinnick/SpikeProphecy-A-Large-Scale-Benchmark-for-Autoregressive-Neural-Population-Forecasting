"""Run inference for all 7 architectures on Steinmetz session 4.

Writes a single NPZ to S3 containing GT + 7 architecture rate predictions.
Designed to feed Figure 1's multi-arch hero.

Output: <anon>/spike-prophecy/outputs/multi-arch-inference-session4/predictions.npz
"""

import os
import json
import time
from pathlib import Path

import boto3
import numpy as np
import torch

BUCKET = "<lab-bucket>"
S3_OUT = "<anon>/spike-prophecy/outputs/multi-arch-inference-session4/predictions.npz"
LOCAL_CACHE = Path("/data/steinmetz_cache")
SESSION_IDX = 4
HISTORY_BINS = 10

# (architecture key for create_teacher_model, S3 slug for checkpoint, display name)
ARCHS = [
    ("mamba",       "2026-03-26_baseline-mamba-v12",                  "Mamba"),
    ("hgrn2",       "2026-04-21_2026-04-21_baseline-hgrn2-v1",        "HGRN2"),
    ("transformer", "2026-03-25_baseline-transformer-v12",            "Transformer"),
    ("gated_delta", "2026-04-22_2026-04-22_baseline-gated-delta-v1",  "GatedDeltaNet"),
    ("lru",         "2026-03-25_baseline-lru-v12",                    "LRU"),
    ("lstm",        "2026-04-15_baseline-lstm-v23",                   "LSTM"),
]
# SNN is a student, different code path — pull from existing cached array
SNN_CACHE_KEY = (
    "<anon>/spike-prophecy/outputs/full-inference-arrays/session_004.npz"
)

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get("ENDPOINT",
        os.environ.get("S3_ENDPOINT", "https://s3-west.nrp-nautilus.io")),
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)


def download_session_cache():
    LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
    prefix = "<anon>/spike-prophecy/inputs/steinmetz-session-cache/"
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, MaxKeys=500)
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        fname = key.split("/")[-1]
        local = LOCAL_CACHE / fname
        # Only fetch session_004 + metadata to keep it fast
        if "session_004" not in fname and fname != "metadata.json":
            continue
        if not local.exists():
            print(f"  fetching {fname}")
            s3.download_file(BUCKET, key, str(local))
    return LOCAL_CACHE


def window_session(data, T):
    T_total, M = data.shape
    n = T_total - T
    X = np.zeros((n, T, M), dtype=np.float32)
    y = np.zeros((n, M), dtype=np.float32)
    for i in range(n):
        X[i] = data[i:i + T]
        y[i] = data[i + T]
    return X, y


def run_inference(model, X, batch=256, device="cuda"):
    model.eval()
    out = np.zeros((X.shape[0], X.shape[2]), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device)
            # session-mask: all neurons real
            mask = torch.ones(xb.shape[0], xb.shape[2], device=device)
            try:
                pred = model(xb, mask=mask)
            except TypeError:
                pred = model(xb)
            if isinstance(pred, dict):
                pred = pred.get("rate", pred.get("output", pred))
            out[i:i + batch] = pred.detach().float().cpu().numpy()
    return out


def main():
    cache = download_session_cache()
    metadata = json.load(open(cache / "metadata.json"))
    sess_info = metadata["sessions"][SESSION_IDX]
    n_neurons = sess_info.get("num_units", sess_info.get("n_units"))
    npy = cache / sess_info.get("npy", f"session_{SESSION_IDX:03d}.npy")
    raw = np.load(str(npy)).astype(np.float32)
    data = raw[:, :n_neurons]
    print(f"Session {SESSION_IDX}: T={data.shape[0]}, M={n_neurons}")

    X, y = window_session(data, HISTORY_BINS)
    print(f"Windowed: X={X.shape}, y={y.shape}")

    out = {"gt": y, "session_idx": np.int64(SESSION_IDX),
           "m_actual": np.int64(n_neurons)}

    # Lazy import (model code path)
    import sys
    sys.path.insert(0, "/workspace")
    from src.models.common import create_teacher_model

    for arch_key, slug, name in ARCHS:
        print(f"\n--- {name} ({arch_key}) ---")
        ckpt_key = f"<anon>/spike-prophecy/outputs/{slug}/best_model.pt"
        ckpt_path = LOCAL_CACHE / f"ckpt_{arch_key}.pt"
        if not ckpt_path.exists():
            print(f"  downloading {ckpt_key}")
            s3.download_file(BUCKET, ckpt_key, str(ckpt_path))
        # Load checkpoint, build model, run inference
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        config = ckpt.get("config") or ckpt.get("teacher_config") or {}
        # Some checkpoints store the model state under different keys
        state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
        # Inject input size + session dims
        try:
            session_dims = {f"session_{SESSION_IDX:03d}": n_neurons}
            model = create_teacher_model(config, n_neurons, session_dims)
            model.load_state_dict(state, strict=False)
            model = model.cuda()
            t0 = time.time()
            preds = run_inference(model, X, device="cuda")
            print(f"  {name}: shape={preds.shape}, took {time.time()-t0:.1f}s")
            out[f"{arch_key}_rates"] = preds
        except Exception as e:
            print(f"  {name}: FAILED ({type(e).__name__}: {e})")

    # SNN: pull from existing cache
    print("\n--- SNN (from cache) ---")
    cache_path = LOCAL_CACHE / "snn_cache.npz"
    s3.download_file(BUCKET, SNN_CACHE_KEY, str(cache_path))
    snn_cache = np.load(str(cache_path))
    out["snn_rates"] = snn_cache["snn_rates"]
    print(f"  shape={snn_cache['snn_rates'].shape}")

    # Save and upload
    local_npz = LOCAL_CACHE / "predictions.npz"
    np.savez_compressed(str(local_npz), **out)
    print(f"\nLocal: {local_npz} ({local_npz.stat().st_size:,} bytes)")
    s3.upload_file(str(local_npz), BUCKET, S3_OUT)
    print(f"Uploaded: s3://{BUCKET}/{S3_OUT}")


if __name__ == "__main__":
    main()
