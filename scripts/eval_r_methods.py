"""Evaluate all distill checkpoints with 3 r methods on NRP.

Downloads session caches from S3 before running inference.
Also evaluates standalone SNN for comparison.
"""
import sys, json, torch, numpy as np, boto3, os, shutil
from pathlib import Path

sys.path.insert(0, "/workspace")
from src.utils.device import resolve_device
from src.utils.config import load_config
from src.models.student import StudentSNN
from src.data.multi_session_loader import create_dataloaders

device = resolve_device()
print(f"Device: {device}")

# S3 client
s3 = boto3.client("s3",
    endpoint_url=os.environ.get("ENDPOINT", os.environ.get("S3_ENDPOINT", "https://s3-west.nrp-nautilus.io")),
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"])

# Teacher baselines (from their training metrics.json)
TEACHER_R = {"steinmetz": 0.4989, "ibl": 0.5532, "combined": 0.5561}

# --- Helper: download S3 cache directory ---
def download_s3_cache(s3_prefix, local_dir):
    """Download all files from an S3 prefix to a local directory."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    resp = s3.list_objects_v2(Bucket="braingeneersdev", Prefix=s3_prefix, MaxKeys=500)
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        fname = key.split("/")[-1]
        local_path = local_dir / fname
        if not local_path.exists():
            print(f"  Downloading {fname} ({obj['Size']:,} bytes)...")
            s3.download_file("braingeneersdev", key, str(local_path))
    print(f"  Cache ready: {local_dir} ({len(list(local_dir.iterdir()))} files)")
    return local_dir

# --- Runs to evaluate ---
RUNS = [
    # Distilled models
    {"name": "Steinmetz Distill v6", "key": "steinmetz",
     "s3_ckpt": "2026-04-06_masked-distill-steinmetz-v6",
     "s3_cache": "<anon>/spike-prophecy/inputs/steinmetz-session-cache",
     "cache_local": "/data/steinmetz_cache",
     "m_max": 1240, "nl": 3},
    {"name": "IBL Distill v6", "key": "ibl",
     "s3_ckpt": "2026-04-06_masked-distill-ibl-v6",
     "s3_cache": "<anon>/spike-prophecy/inputs/ibl-repeated-site",
     "cache_local": "/data/ibl_cache",
     "m_max": 1998, "nl": 3},
    {"name": "Combined Distill v7", "key": "combined",
     "s3_ckpt": "2026-04-07_masked-distill-combined-v7",
     "s3_cache": "<anon>/spike-prophecy/inputs/combined-steinmetz-ibl",
     "cache_local": "/data/combined_cache",
     "m_max": 1998, "nl": 3},
    # Standalone SNN (no distillation)
    {"name": "Standalone SNN v12b", "key": "steinmetz",
     "s3_ckpt": "snn-standalone-v12b",
     "s3_cache": "<anon>/spike-prophecy/inputs/steinmetz-session-cache",
     "cache_local": "/data/steinmetz_cache",
     "m_max": 1240, "nl": 2},
]


def compute_all_r(preds_list, targets_list, m):
    """Compute 3 r methods using streaming Welford stats."""
    sx = np.zeros(m, dtype=np.float64)
    sy = np.zeros(m, dtype=np.float64)
    sxy = np.zeros(m, dtype=np.float64)
    sx2 = np.zeros(m, dtype=np.float64)
    sy2 = np.zeros(m, dtype=np.float64)
    gsx = gsy = gsxy = gsx2 = gsy2 = np.float64(0)
    N = 0

    for preds, targets in zip(preds_list, targets_list):
        p = preds.numpy().astype(np.float64)
        t = targets.numpy().astype(np.float64)
        sx += p.sum(0); sy += t.sum(0)
        sxy += (p * t).sum(0)
        sx2 += (p ** 2).sum(0); sy2 += (t ** 2).sum(0)
        gsx += p.sum(); gsy += t.sum()
        gsxy += (p * t).sum()
        gsx2 += (p ** 2).sum(); gsy2 += (t ** 2).sum()
        N += p.shape[0]

    Nf = np.float64(N)
    num = Nf * sxy - sx * sy
    den = np.sqrt((Nf * sx2 - sx ** 2) * (Nf * sy2 - sy ** 2))
    ch_r = np.where(den > 1e-12, num / den, 0.0)

    per_ch = float(np.nanmean(ch_r))
    w = sy; w = w / (w.sum() + 1e-8)
    weighted = float(np.nansum(ch_r * w))
    G = Nf * m
    gn = G * gsxy - gsx * gsy
    gd = np.sqrt((G * gsx2 - gsx ** 2) * (G * gsy2 - gsy ** 2))
    global_r = float(gn / gd) if gd > 1e-12 else 0.0

    return {"per_ch": per_ch, "weighted": weighted, "global": global_r, "N": N}


# --- Main loop ---
results = []
for run in RUNS:
    print(f"\n{'='*60}")
    print(f"  {run['name']}")
    print(f"{'='*60}")

    # Download checkpoint from S3
    ckpt_path = f"/tmp/{run['s3_ckpt']}_best.pt"
    if not os.path.exists(ckpt_path):
        print(f"  Downloading checkpoint from {run['s3_ckpt']}...")
        s3.download_file("braingeneersdev",
            f"<anon>/spike-prophecy/outputs/{run['s3_ckpt']}/best_model.pt",
            ckpt_path)

    # Download session cache from S3
    cache_dir = download_s3_cache(run["s3_cache"], run["cache_local"])

    # Load metadata and create dataloaders
    metadata = json.load(open(cache_dir / "metadata.json"))
    print(f"  m_max={metadata['m_max']}, n_sessions={metadata.get('n_sessions','?')}")

    # Create a minimal data config for create_dataloaders
    data_cfg = {
        "training": {"batch_size": 256, "val_split": 0.1, "test_split": 0.1},
        "seed": 42,
    }
    loaders = create_dataloaders(cache_dir, metadata, data_cfg)
    test_loader = loaders["test"]

    # Load model
    m_max = run["m_max"]
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model = StudentSNN(
        input_size=m_max, output_size=m_max, hidden_size=256,
        num_layers=run["nl"], neuron_type="rsynaptic", beta=0.9,
        threshold=1.0, gradient_slope=25.0, learn_beta=True,
        alpha=0.85, dropout=0.0)
    model.load_state_dict(sd)
    model.to(device).eval()
    print(f"  Model loaded: epoch={ckpt.get('epoch','?')}, layers={run['nl']}")

    # Run inference (streaming — no full concat)
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            y = batch[1]
            out = model(x)
            rates = (out[0] if isinstance(out, tuple) else out).cpu()
            all_preds.append(rates)
            all_targets.append(y)

    # Compute all 3 r methods
    r_vals = compute_all_r(all_preds, all_targets, m_max)
    teacher = TEACHER_R[run["key"]]

    print(f"  N={r_vals['N']:,} samples, Teacher r={teacher:.4f}")
    print(f"  Per-channel r:       {r_vals['per_ch']:.4f}  ({r_vals['per_ch']/teacher*100:.1f}% retention)")
    print(f"  Activity-weighted r: {r_vals['weighted']:.4f}  ({r_vals['weighted']/teacher*100:.1f}% retention)")
    print(f"  Global flatten r:    {r_vals['global']:.4f}  ({r_vals['global']/teacher*100:.1f}% retention)")

    results.append({"name": run["name"], "key": run["key"], "teacher": teacher,
                     "layers": run["nl"], **r_vals})

    # Free memory
    del model, all_preds, all_targets
    torch.cuda.empty_cache()

# --- Summary table ---
print(f"\n{'='*80}")
print(f"  FINAL RESULTS — 3 Pearson r Methods")
print(f"{'='*80}")
print(f"{'Model':<26} {'Layers':>6} {'Teacher':>8} {'PerCh':>8} {'Weighted':>8} {'Global':>8} {'Ret%':>6}")
print("-" * 80)
for r in results:
    ret = r["weighted"] / r["teacher"] * 100
    print(f"  {r['name']:<24} {r['layers']:>6} {r['teacher']:>8.4f} "
          f"{r['per_ch']:>8.4f} {r['weighted']:>8.4f} {r['global']:>8.4f} {ret:>5.1f}%")

# Upload results to S3
out = json.dumps(results, indent=2)
s3.put_object(Bucket="braingeneersdev",
    Key="<anon>/spike-prophecy/outputs/eval-distill-r-methods/results.json",
    Body=out.encode())
print(f"\nUploaded results to S3: eval-distill-r-methods/results.json")
