import json
import logging
import numpy as np
from pathlib import Path
import yaml
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    cache_path = PROJECT_ROOT / "data" / "processed" / "ibl_repeated_site_cache"
    metadata_out = cache_path / "metadata.json"
    
    # Load config splits
    config_file = PROJECT_ROOT / "configs" / "data" / "ibl_repeated_site_full_nrp.yaml"
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    splits = config.get("splits", {"train": 0.7, "val": 0.15, "test": 0.15})
    bin_width_ms = config.get("bin_width_ms", 50.0)
    history_bins = config.get("history_bins", 10)
    
    # Load the matching EIDs the loop was crunching (it sorts them based on ONE index)
    from src.data.ibl_data_loader import list_ibl_sessions
    one_cache_dir = config.get("ibl", {}).get("cache_dir", "data/raw/ibl")
    all_eids = list_ibl_sessions(tag="repeated_site", cache_dir=one_cache_dir)
    
    # Analyze the surviving npys
    npy_files = sorted(list(cache_path.glob("session_*.npy")))
    n_sessions = len(npy_files)
    
    logging.info(f"Discovered {n_sessions} cached arrays. Generating structural metadata.")
    
    session_info = []
    
    for i, npy_path in enumerate(npy_files):
        eid = all_eids[i]
        counts = np.load(npy_path)
        m_i, t_i = counts.shape
        
        train_end = int(t_i * splits["train"])
        val_end = train_end + int(t_i * splits["val"])
        
        session_info.append({
            "index": i,
            "file": eid,
            "npy_file": f"data\\processed\\ibl_repeated_site_cache\\{npy_path.name}",
            "num_units": m_i,
            "num_bins": t_i,
            "duration_s": round(t_i * bin_width_ms / 1000, 2),
            "split_boundaries": {
                "train_end": train_end,
                "val_end": val_end,
            },
            "brain_regions": None,
        })
        logging.info(f"  Mapped {npy_path.name} to {eid} (M: {m_i}, T: {t_i})")

    all_m = [s["num_units"] for s in session_info]
    m_max = int(max(all_m))
    
    # Structure metadata.json exactly as multi_session_loader expects
    final_metadata = {
        "num_sessions": n_sessions,
        "m_max": m_max,
        "history_bins": history_bins,
        "bin_width_ms": bin_width_ms,
        "n_features_per_channel": 0,
        "n_covariates": 0,
        "covariate_names": [],
        "sessions": session_info
    }
    
    with open(metadata_out, "w", encoding="utf-8") as f:
        json.dump(final_metadata, f, indent=2)
        
    logging.info(f"Successfully generated full metadata.json for {n_sessions} sessions with M_max={m_max}.")

if __name__ == "__main__":
    main()
