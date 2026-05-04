import yaml
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ibl_data_loader import preprocess_and_cache_ibl

def main():
    config_path = PROJECT_ROOT / "configs" / "data" / "ibl_repeated_site_full_nrp.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    # Create the cache using the same path as expected by the S3 scripts
    ibl_cache_dir = "data/processed/ibl_repeated_site_cache"
    print(f"Preprocessing {config['ibl']['max_sessions']} IBL sessions into cache: {ibl_cache_dir}")
    cache_path, metadata = preprocess_and_cache_ibl(
        config, cache_dir=ibl_cache_dir, force_reprocess=True
    )
    print(f"\nDone! Extracted {metadata['num_sessions']} sessions into {cache_path}")
    print(f"M_max padding size: {metadata['m_max']}")

if __name__ == "__main__":
    main()
