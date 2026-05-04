"""Pull NRP-computed 7-arch data and emit updated FANO_DATA / AR_ROLLOUT
blocks for scripts/figures/data.py.

Usage:
  python scripts/figures/_update_data_from_nrp.py

Reads:
  s3://braingeneersdev/<anon>/spike-prophecy/outputs/per-neuron-fano-7arch/per_neuron_data.npz
  s3://braingeneersdev/<anon>/spike-prophecy/outputs/ar-rollout-7arch/ar_rollout.json

Writes (printed to stdout for manual paste-in):
  - FANO_DATA dict with all 7 archs
  - AR_ROLLOUT dict in multi-arch form
"""

import json
import os
from pathlib import Path

import boto3
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "figure_cache"
CACHE.mkdir(parents=True, exist_ok=True)

S3_FANO = "<anon>/spike-prophecy/outputs/per-neuron-fano-7arch/per_neuron_data.npz"
S3_AR = "<anon>/spike-prophecy/outputs/ar-rollout-7arch/ar_rollout.json"

s3 = boto3.client(
    "s3",
    endpoint_url="https://s3-west.nrp-nautilus.io",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)


def fetch(key, dest):
    s3.download_file("braingeneersdev", key, str(dest))
    return dest


def compute_fano_data():
    """Compute FANO_DATA dict from per_neuron_data.npz."""
    npz_path = CACHE / "per_neuron_data.npz"
    fetch(S3_FANO, npz_path)
    d = np.load(str(npz_path), allow_pickle=True)
    arch_names = list(d["arch_names"])
    fano_per_session = d["fano_per_session"]
    n_sessions = len(fano_per_session)
    print(f"Loaded {n_sessions} sessions, {len(arch_names)} archs: "
          f"{arch_names}")

    # Aggregate Fano + per-neuron-r across all sessions for each arch.
    bin_edges = [0.0, 0.8, 1.0, 1.2, 1.5, np.inf]
    bin_labels = ["FF<0.8", "0.8<=FF<1.0", "1.0<=FF<1.2",
                  "1.2<=FF<1.5", "FF>=1.5"]

    # Concatenate Fano across sessions
    all_fano = np.concatenate([f for f in fano_per_session
                               if f is not None and len(f) > 0])
    valid = ~np.isnan(all_fano)
    print(f"  Total neurons across sessions: {valid.sum()}")
    bin_idx_all = np.digitize(all_fano, bin_edges) - 1

    fano_data = {}
    for name in arch_names:
        pn_per_session = d[f"pn_r__{name}"]
        all_pn = np.concatenate([p for p in pn_per_session
                                 if p is not None and len(p) > 0])
        if len(all_pn) != len(all_fano):
            print(f"  WARN {name}: {len(all_pn)} pn vs {len(all_fano)} fano")
            continue
        per_bin = []
        for i in range(len(bin_labels)):
            mask = (bin_idx_all == i) & valid & ~np.isnan(all_pn)
            if mask.sum() > 0:
                per_bin.append(round(float(np.mean(all_pn[mask])), 3))
            else:
                per_bin.append(0.0)
        fano_data[name] = per_bin

    return fano_data, bin_labels


def compute_ar_rollout():
    """Pull ar_rollout.json from S3."""
    out_path = CACHE / "ar_rollout.json"
    fetch(S3_AR, out_path)
    return json.load(open(out_path))


def main():
    print("=== Computing FANO_DATA ===")
    try:
        fano_data, bin_labels = compute_fano_data()
        print("\nFANO_DATA = {")
        for name, vals in fano_data.items():
            vals_str = ", ".join(f"{v:.3f}" for v in vals)
            print(f"    '{name}': [{vals_str}],")
        print("}")
        print(f"\nFANO_BINS = {bin_labels}")
    except Exception as e:
        print(f"  Fano fetch/compute failed: {e}")
        fano_data = None

    print("\n=== Computing AR_ROLLOUT ===")
    try:
        ar = compute_ar_rollout()
        # NRP-side script writes "k_steps"; figure expects "steps".
        steps = ar.get("steps") or ar.get("k_steps")
        print("AR_ROLLOUT = {")
        print(f"    'steps': {steps},")
        print("    'archs': {")
        for name, d in ar["archs"].items():
            pop_r_str = ", ".join(f"{v:.3f}" for v in d["pop_r"])
            n_r_str = ", ".join(f"{v:.3f}" for v in d["neuron_r"])
            print(f"        '{name}': {{")
            print(f"            'pop_r': [{pop_r_str}],")
            print(f"            'neuron_r': [{n_r_str}],")
            print(f"        }},")
        print("    },")
        print("}")
        # Normalise into a single shape for downstream consumption.
        ar = {"steps": steps, "archs": ar["archs"]}
    except Exception as e:
        print(f"  AR rollout fetch failed: {e}")
        ar = None

    # Save the dict for direct programmatic pickup
    if fano_data is not None or ar is not None:
        out = {"FANO_DATA": fano_data, "AR_ROLLOUT": ar}
        with open(CACHE / "nrp_7arch_data.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote: {CACHE / 'nrp_7arch_data.json'}")


if __name__ == "__main__":
    main()
