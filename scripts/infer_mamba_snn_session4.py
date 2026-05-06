"""Run Mamba + SNN inference on Steinmetz session 4 (NRP-only).

Uses the verified-correct pipeline from the local run:
  - Transpose npy from (n_neurons, n_bins) -> (T_total, n_neurons)
  - Zero-pad to M_max=1240 (Steinmetz training-time width)
  - No session_dims (use shared input/output projections)
  - Trim outputs to actual n_neurons + first 660 timesteps for the figure

Output:
  s3://<lab-bucket>/<anon>/spike-prophecy/outputs/multi-arch-mamba-snn-session4/predictions.npz
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

sys.path.insert(0, "/workspace")

BUCKET = "<lab-bucket>"
S3_OUT = ("<anon>/spike-prophecy/outputs/"
          "multi-arch-mamba-snn-session4/predictions.npz")
LOCAL = Path("/data/work")
LOCAL.mkdir(parents=True, exist_ok=True)
SESSION_IDX = 4
HISTORY_BINS = 10
M_TRAIN = 1240
DISPLAY_LEN = 660

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get("ENDPOINT",
        os.environ.get("S3_ENDPOINT", "https://s3-west.nrp-nautilus.io")),
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)


def fetch(key, dest):
    if dest.exists():
        return dest
    print(f"  fetching {key.split('/')[-1]}")
    s3.download_file(BUCKET, key, str(dest))
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
    """SNN models return (rate, spikes) tuples; teachers return tensor or
    dict.  Reduce to a single rate tensor."""
    if isinstance(pred, dict):
        for k in ("rate", "rates", "output", "out"):
            if k in pred:
                return pred[k]
        return list(pred.values())[0]
    if isinstance(pred, (list, tuple)):
        # Heuristic: first element is usually the rate output
        return pred[0]
    return pred


def run_inference(model, X, n_actual, batch=128, device="cuda"):
    model.eval()
    out = np.zeros((X.shape[0], n_actual), dtype=np.float32)
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
    return out


def main():
    # Load session
    fetch("<anon>/spike-prophecy/inputs/steinmetz-session-cache/metadata.json",
          LOCAL / "metadata.json")
    md = json.load(open(LOCAL / "metadata.json"))
    sess = md["sessions"][SESSION_IDX]
    n_actual = sess.get("num_units")
    n_bins = sess.get("num_bins")
    npy = sess.get("npy", f"session_{SESSION_IDX:03d}.npy")
    fetch(f"<anon>/spike-prophecy/inputs/steinmetz-session-cache/{npy}",
          LOCAL / npy)
    raw = np.load(str(LOCAL / npy)).astype(np.float32)
    if raw.shape[0] == n_actual and raw.shape[1] == n_bins:
        raw = raw.T
    print(f"raw transposed: {raw.shape}; n_actual={n_actual}")
    # Zero-pad to M_TRAIN
    raw_padded = np.zeros((raw.shape[0], M_TRAIN), dtype=np.float32)
    raw_padded[:, :n_actual] = raw[:, :n_actual]
    X, y = window(raw_padded, HISTORY_BINS)
    print(f"windowed: X={X.shape}")
    y_actual = y[:, :n_actual]

    out = {"gt": y_actual[:DISPLAY_LEN].astype(np.float32),
           "session_idx": np.int64(SESSION_IDX),
           "m_actual": np.int64(n_actual)}

    # ----------- Mamba -----------
    print("\n--- Mamba ---")
    from src.models.common import create_teacher_model
    mamba_yaml = "/workspace/configs/teacher/nrp_teacher_mamba.yaml"
    mamba_ckpt_key = ("<anon>/spike-prophecy/outputs/"
                      "2026-03-26_baseline-mamba-v12/best_model.pt")
    fetch(mamba_ckpt_key, LOCAL / "ckpt_mamba.pt")
    config = yaml.safe_load(open(mamba_yaml))
    config.setdefault("model", {})["architecture"] = "mamba"
    try:
        ckpt = torch.load(str(LOCAL / "ckpt_mamba.pt"),
                          map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"]
        model = create_teacher_model(config, M_TRAIN, session_dims=None)
        miss, unexp = model.load_state_dict(state, strict=False)
        if miss:
            print(f"  WARN missing: {len(miss)} (e.g. {miss[:2]})")
        if unexp:
            print(f"  WARN unexpected: {len(unexp)} (e.g. {unexp[:2]})")
        model = model.cuda()
        t0 = time.time()
        preds = run_inference(model, X, n_actual)
        out["mamba_rates"] = preds[:DISPLAY_LEN].astype(np.float32)
        print(f"  Mamba: shape={preds.shape}, {time.time()-t0:.1f}s")
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Mamba: FAILED ({type(e).__name__}: {e})")

    # ----------- SNN (StudentSNN) -----------
    print("\n--- SNN (3L standalone) ---")
    from src.models.student import StudentSNN
    snn_yaml = "/workspace/configs/student/standalone_snn_3l.yaml"
    snn_ckpt_key = ("<anon>/spike-prophecy/outputs/"
                    "2026-04-22_snn-standalone-3l-steinmetz/best_model.pt")
    if not Path(snn_yaml).exists():
        # Fall back to S3 if not in image
        fetch("<anon>/spike-prophecy/scripts/standalone_snn_3l.yaml",
              LOCAL / "standalone_snn_3l.yaml")
        snn_yaml = str(LOCAL / "standalone_snn_3l.yaml")
    fetch(snn_ckpt_key, LOCAL / "ckpt_snn.pt")
    config = yaml.safe_load(open(snn_yaml))
    try:
        ckpt = torch.load(str(LOCAL / "ckpt_snn.pt"),
                          map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
        model = StudentSNN.from_config(config, M_TRAIN)
        miss, unexp = model.load_state_dict(state, strict=False)
        if miss:
            print(f"  WARN missing: {len(miss)} (e.g. {miss[:2]})")
        if unexp:
            print(f"  WARN unexpected: {len(unexp)} (e.g. {unexp[:2]})")
        model = model.cuda()
        t0 = time.time()
        preds = run_inference(model, X, n_actual)
        out["snn_rates"] = preds[:DISPLAY_LEN].astype(np.float32)
        print(f"  SNN: shape={preds.shape}, {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"  SNN: FAILED ({type(e).__name__}: {e})")

    # Save and upload
    local_npz = LOCAL / "mamba_snn_predictions.npz"
    np.savez_compressed(str(local_npz), **out)
    print(f"\nLocal: {local_npz} ({local_npz.stat().st_size:,} bytes)")
    print(f"Keys: {list(out.keys())}")
    s3.upload_file(str(local_npz), BUCKET, S3_OUT)
    print(f"Uploaded: s3://{BUCKET}/{S3_OUT}")


if __name__ == "__main__":
    main()
