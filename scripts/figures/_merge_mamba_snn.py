"""Merge Mamba and SNN rates into the local multi-arch NPZ.

Sources (best-effort, in order):
  - data/figure_cache/_inference_workdir/mamba_snn_predictions.npz
    (NRP run; supplies mamba_rates and possibly snn_rates)
  - data/figure_cache/_inference_workdir/snn_session4_local.npz
    (local Windows SNN run; supplies snn_rates if NRP missed it)

Target:
  - data/figure_cache/multi_arch_session4.npz (overwrite with all available)
"""

from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
NRP = ROOT / "data" / "figure_cache" / "_inference_workdir" / "mamba_snn_predictions.npz"
SNN_LOCAL = ROOT / "data" / "figure_cache" / "_inference_workdir" / "snn_session4_local.npz"
LOCAL = ROOT / "data" / "figure_cache" / "multi_arch_session4.npz"


def _trim_to(rates, T_local, n_local):
    if rates.shape[0] > T_local:
        rates = rates[:T_local]
    if rates.shape[1] > n_local:
        rates = rates[:, :n_local]
    return rates.astype(np.float32)


def main():
    local = np.load(str(LOCAL))
    print("Local multi-arch NPZ keys:", list(local.files))
    merged = {k: local[k] for k in local.files}
    gt = local["gt"]
    T_local, n_local = gt.shape
    print(f"Local gt: {gt.shape}, n_actual={n_local}")

    # NRP source (Mamba + maybe SNN)
    if NRP.exists():
        nrp = np.load(str(NRP))
        print(f"\nNRP NPZ keys: {list(nrp.files)}")
        for src_key in ["mamba_rates", "snn_rates"]:
            if src_key in nrp.files:
                rates = _trim_to(nrp[src_key], T_local, n_local)
                merged[src_key] = rates
                r = pearsonr(gt[:rates.shape[0]].sum(axis=1),
                             rates.sum(axis=1))[0]
                print(f"  [NRP] merged {src_key}: shape={rates.shape}, "
                      f"pop_r={r:.4f}")
            else:
                print(f"  [NRP] {src_key} missing")
    else:
        print(f"NRP NPZ missing at {NRP}")

    # Local SNN fallback (only if SNN didn't come from NRP)
    if "snn_rates" not in merged and SNN_LOCAL.exists():
        snn = np.load(str(SNN_LOCAL))
        if "snn_rates" in snn.files:
            rates = _trim_to(snn["snn_rates"], T_local, n_local)
            merged["snn_rates"] = rates
            r = pearsonr(gt[:rates.shape[0]].sum(axis=1),
                         rates.sum(axis=1))[0]
            print(f"  [LOCAL] merged snn_rates: shape={rates.shape}, "
                  f"pop_r={r:.4f}")

    # Compute pop_r for the existing 5 ANN archs as a sanity check
    print("\nPop_r summary (vs local gt):")
    for k in ["mamba_rates", "hgrn2_rates", "transformer_rates",
              "gated_delta_rates", "lru_rates", "lstm_rates", "snn_rates"]:
        if k not in merged:
            continue
        r = pearsonr(gt.sum(axis=1)[:merged[k].shape[0]],
                     merged[k].sum(axis=1))[0]
        print(f"  {k:24s} pop_r = {r:.4f}")

    np.savez_compressed(str(LOCAL), **merged)
    print(f"\nSaved merged NPZ: {LOCAL} ({LOCAL.stat().st_size:,} bytes)")
    print(f"Keys: {sorted(merged.keys())}")


if __name__ == "__main__":
    main()
