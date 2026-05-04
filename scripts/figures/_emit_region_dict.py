"""Print REGION_DATA_BY_ARCH in data.py-pasteable form."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "data" / "figure_cache" / "region_per_arch.json") as f:
    d = json.load(f)

arch_order = ["Mamba", "HGRN2", "Transformer", "GatedDelta",
              "LRU", "LSTM", "SNN"]

print("REGION_DATA_BY_ARCH = {")
for arch in arch_order:
    if arch not in d:
        continue
    print(f"    '{arch}': {{")
    for region in sorted(d[arch].keys()):
        v = d[arch][region]
        rkey = repr(region)  # handles \n correctly
        print(f"        {rkey}: "
              f"{{'n': {v['n']:5d}, 'raw': {v['raw']:.3f}, "
              f"'adjusted': {v['adjusted']:.3f}}},")
    print("    },")
print("}")
