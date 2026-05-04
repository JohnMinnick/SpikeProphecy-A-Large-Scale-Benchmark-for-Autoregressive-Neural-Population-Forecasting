"""Explore which per-bin / per-neuron views best differentiate the
architectures on session 4."""

from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "data" / "figure_cache" / "multi_arch_session4.npz"


def main():
    d = np.load(str(NPZ))
    gt = d["gt"]   # (T, N) = (660, 703)
    print(f"GT shape: {gt.shape}, mean={gt.mean():.3f}")

    archs = ["mamba_rates", "hgrn2_rates", "transformer_rates",
             "gated_delta_rates", "lru_rates", "lstm_rates", "snn_rates"]

    print("\nPer-arch metrics on session 4:")
    print(f"{'Arch':14s} {'pop_r':>8s} {'spat_r':>8s} {'cos':>8s}"
          f" {'pn_r_med':>10s} {'pn_r_mean':>10s} {'spat_r_sd':>10s}")
    for k in archs:
        if k not in d.files:
            continue
        rates = d[k]
        n = min(rates.shape[0], gt.shape[0])
        p = rates[:n]
        g = gt[:n]
        # Population r
        pop_r = pearsonr(g.sum(1), p.sum(1))[0]
        # Spatial r (per-time-bin cross-neuron Pearson, then averaged)
        per_time_spat = []
        for t in range(n):
            if g[t].std() > 0 and p[t].std() > 0:
                per_time_spat.append(pearsonr(g[t], p[t])[0])
            else:
                per_time_spat.append(np.nan)
        spat_r = np.nanmean(per_time_spat)
        spat_r_sd = np.nanstd(per_time_spat)
        # Cosine
        cos_sims = []
        for t in range(n):
            ng, np_p = np.linalg.norm(g[t]), np.linalg.norm(p[t])
            if ng > 0 and np_p > 0:
                cos_sims.append(np.dot(g[t], p[t]) / (ng * np_p))
        cos = np.mean(cos_sims)
        # Per-neuron r (active-neuron only)
        pn_rs = []
        for j in range(g.shape[1]):
            if g[:, j].std() > 0 and p[:, j].std() > 0:
                pn_rs.append(pearsonr(g[:, j], p[:, j])[0])
        pn_r_med = np.median(pn_rs)
        pn_r_mean = np.mean(pn_rs)

        print(f"{k:14s} {pop_r:8.3f} {spat_r:8.3f} {cos:8.3f}"
              f" {pn_r_med:10.3f} {pn_r_mean:10.3f} {spat_r_sd:10.3f}")


if __name__ == "__main__":
    main()
