"""Collate distillation ablation runs into a single table for app:distillation.

Walks outputs/s3_metrics/, finds distillation experiments (excludes per-session
twin runs), reads each config + metrics JSON, and tabulates:
  slug | dataset | distill_w | reg_w | num_layers | best_val_r | epoch | notes

Outputs Markdown + LaTeX tables.
"""

import json
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SLUGS_OF_INTEREST = [
    # Aggregate distillation runs (NOT per-session twins)
    "2026-03-05_distill-nocov-baseline",
    "2026-03-26_gac-distill-v12",
    "2026-04-03_distill-3layer-steinmetz-v1",
    "2026-04-03_distill-steinmetz-v3",
    "2026-04-06_masked-distill-ibl-v6",
    "2026-04-06_masked-distill-steinmetz-v6",
    "2026-04-07_masked-distill-combined-v6",
    "distill-beta0-ablation-v2",
    # Multi-head distillation variants (Steinmetz, 16-class)
    "2026-03-23_distill-mamba-multi-head-v5a",
    "2026-03-23_distill-mamba-multi-head-v5b",
    "2026-03-23_distill-mamba-multi-head-v5c",
    "2026-03-23_distill-mamba-multi-head-v5d",
]


def best_val_r(metrics_path: Path) -> tuple:
    """Return (best_val_r, best_epoch, n_epochs) from a metrics.json."""
    if not metrics_path.exists():
        return (None, None, 0)
    try:
        m = json.load(open(metrics_path))
    except Exception:
        return (None, None, 0)
    if isinstance(m, list):
        if not m:
            return (None, None, 0)
        # Per-epoch list
        best_r = -1
        best_e = None
        for e in m:
            if not isinstance(e, dict):
                continue
            r = e.get("val_r") or e.get("val_pearson_r")
            if r is not None and r > best_r:
                best_r = r
                best_e = e.get("epoch")
        return (best_r if best_r > 0 else None, best_e, len(m))
    if isinstance(m, dict):
        # Could be top-level summary
        r = (
            m.get("teacher_best_val_pearson_r")
            or m.get("best_val_pearson_r")
            or m.get("best_val_r")
        )
        e = m.get("teacher_n_epochs_trained") or m.get("n_epochs_trained")
        return (r, e, e or 0)
    return (None, None, 0)


def get(d: dict, *keys, default=None):
    """Safely get nested keys."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        if k not in d:
            return default
        d = d[k]
    return d


def main() -> None:
    rows = []
    for slug in SLUGS_OF_INTEREST:
        d = PROJECT_ROOT / "outputs" / "s3_metrics" / slug
        if not d.exists():
            print(f"  MISSING: {slug}")
            continue
        cfg_path = d / "config.yaml"
        metrics_path = d / "metrics.json"
        notes_path = d / "notes.md"

        cfg = {}
        if cfg_path.exists():
            try:
                cfg = yaml.safe_load(open(cfg_path))
            except Exception as e:
                print(f"  cfg parse failed for {slug}: {e}")

        # Detect dataset
        data_src = get(cfg, "data", "source", "type", default="?")
        ibl_tag = get(cfg, "data", "ibl", "tag", default=None)
        if data_src == "ibl" and ibl_tag == "combined":
            dataset = "Combined 105"
        elif data_src == "ibl":
            dataset = f"IBL ({ibl_tag or '?'})"
        elif data_src in ("nwb_multi", "nwb"):
            dataset = "Steinmetz 39"
        else:
            dataset = "?"

        # Distill weights
        distill_w = (
            get(cfg, "distillation", "distill_weight")
            or get(cfg, "student", "distillation", "distill_weight")
            or get(cfg, "training", "distill_weight")
        )
        reg_w = (
            get(cfg, "distillation", "reg_weight")
            or get(cfg, "student", "distillation", "reg_weight")
        )
        # Architecture
        student_cfg = (
            get(cfg, "model", default=None)
            or get(cfg, "student", "model", default={})
        )
        nl = student_cfg.get("num_layers", "?") if student_cfg else "?"
        hs = student_cfg.get("hidden_size", "?") if student_cfg else "?"
        nt = student_cfg.get("neuron_type", "?") if student_cfg else "?"
        # Aux heads
        aux = student_cfg.get("auxiliary_heads") if student_cfg else None

        best_r, best_e, n_ep = best_val_r(metrics_path)

        # Notes (first non-empty line)
        note = ""
        if notes_path.exists():
            for line in open(notes_path):
                line = line.strip()
                if line and not line.startswith("#"):
                    note = line[:80]
                    break

        rows.append({
            "slug": slug,
            "dataset": dataset,
            "layers": nl,
            "hidden": hs,
            "neuron": nt,
            "distill_w": distill_w,
            "reg_w": reg_w,
            "aux": aux,
            "best_val_r": best_r,
            "best_epoch": best_e,
            "n_epochs": n_ep,
            "note": note,
        })

    # Sort by date prefix
    rows.sort(key=lambda r: r["slug"])

    # Print Markdown table
    print()
    print("# Distillation ablation summary")
    print()
    print(
        "| Slug | Dataset | L | Hidden | Neuron | beta (distill) | "
        "alpha (reg) | Aux | Best val_r | Epoch |"
    )
    print(
        "|------|---------|---|--------|--------|-------------|"
        "---------|-----|------------|-------|"
    )
    for r in rows:
        bvr = f"{r['best_val_r']:.4f}" if r["best_val_r"] is not None else "—"
        epoch = r["best_epoch"] or "—"
        dw = (
            f"{r['distill_w']:.2f}"
            if isinstance(r["distill_w"], (int, float))
            else "—"
        )
        rw = (
            f"{r['reg_w']:.4f}"
            if isinstance(r["reg_w"], (int, float))
            else "—"
        )
        aux = (
            "+".join(r["aux"])[:10]
            if isinstance(r["aux"], list) and r["aux"]
            else "—"
        )
        print(
            f"| {r['slug'][:38]} | {r['dataset']} | {r['layers']} | "
            f"{r['hidden']} | {r['neuron']} | {dw} | {rw} | {aux} | "
            f"{bvr} | {epoch} |"
        )

    # Save JSON for downstream LaTeX building
    out = PROJECT_ROOT / "outputs" / "eval_local" / "distillation_table.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
