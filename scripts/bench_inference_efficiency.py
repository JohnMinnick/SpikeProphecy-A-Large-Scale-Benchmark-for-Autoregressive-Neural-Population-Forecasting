"""Inference efficiency benchmark for all 5 SpikeProphecy architectures.

Measures inference latency, peak VRAM, and throughput on a single consistent
GPU (NRP RTX 3090) for fair cross-architecture comparison in Table 4 of the
NeurIPS E&D paper.

Config (matches paper body):
    batch size B = 512
    history T = 10 bins
    neurons M = 1240 (Steinmetz max)
    warmup = 20 iterations, timed = 50 iterations

Architectures:
    Mamba (d=256, L=3, d_state=16, expand=2)
    Transformer (d=256, heads=8, L=3, pre-norm)
    LRU v2 (d=256, L=3)
    LSTM (H=256, L=3, dropout=0.2)
    SNN 1L (h=256, RSynaptic, beta=0.9)

Uploads results to s3://<lab-bucket>/<anon>/spike-prophecy/outputs/
    efficiency-benchmark/metrics.json
"""

import json
import os
import sys
import time

import boto3
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/workspace")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BATCH_SIZE = 512
HISTORY_BINS = 10      # T
NUM_NEURONS = 1240     # M (Steinmetz max, matches existing Table 4 row)
WARMUP = 20
TIMED = 50

BUCKET = "<lab-bucket>"
S3_OUTPUT_KEY = "<anon>/spike-prophecy/outputs/efficiency-benchmark/metrics.json"

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get(
        "ENDPOINT",
        os.environ.get("S3_ENDPOINT", "https://s3-west.nrp-nautilus.io"),
    ),
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)


def gpu_name():
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "unknown"


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def benchmark(model: nn.Module, x: torch.Tensor, name: str) -> dict:
    """Measure latency (ms), peak VRAM (MB), throughput (samples/s)."""
    model.eval()
    torch.cuda.synchronize()

    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP):
            out = model(x)
            _ = out[0] if isinstance(out, tuple) else out
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Time
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(TIMED):
            out = model(x)
            _ = out[0] if isinstance(out, tuple) else out
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    latency_ms = (t1 - t0) / TIMED * 1000.0
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    throughput = BATCH_SIZE / (latency_ms / 1000.0)

    return {
        "name": name,
        "n_params": count_params(model),
        "latency_ms": round(latency_ms, 3),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "throughput_samples_per_s": round(throughput, 0),
    }


def build_mamba(M: int) -> nn.Module:
    from src.models.mamba_baseline import TeacherMamba
    cfg = {
        "model": {
            "architecture": "mamba",
            "input_size": M, "hidden_size": 256,
            "num_layers": 3, "d_state": 16, "d_conv": 4, "expand": 2,
            "dropout": 0.0, "use_layer_norm": True,
            "use_attention": False, "use_population_coupling": False,
        },
    }
    return TeacherMamba.from_config(cfg, input_size=M)


def build_transformer(M: int) -> nn.Module:
    from src.models.transformer_baseline import TeacherTransformer
    cfg = {
        "model": {
            "architecture": "transformer",
            "input_size": M, "hidden_size": 256,
            "num_layers": 3, "num_heads": 8,
            "dropout": 0.0, "use_layer_norm": True,
            "use_attention": False, "use_population_coupling": False,
        },
    }
    return TeacherTransformer.from_config(cfg, input_size=M)


def build_lru(M: int) -> nn.Module:
    from src.models.lru import TeacherLRU
    cfg = {
        "model": {
            "architecture": "lru",
            "input_size": M, "hidden_size": 256,
            "num_layers": 3,
            "dropout": 0.0, "use_layer_norm": True,
            "use_attention": False, "use_population_coupling": False,
        },
    }
    return TeacherLRU.from_config(cfg, input_size=M)


def build_lstm(M: int) -> nn.Module:
    from src.models.teacher import TeacherLSTM
    cfg = {
        "model": {
            "architecture": "lstm",
            "input_size": M, "hidden_size": 256,
            "num_layers": 3,
            "dropout": 0.2, "use_layer_norm": True,
            "use_attention": False, "use_population_coupling": False,
        },
    }
    return TeacherLSTM.from_config(cfg, input_size=M)


def build_snn_1l(M: int) -> nn.Module:
    from src.models.student import StudentSNN
    return StudentSNN(
        input_size=M, output_size=M, hidden_size=256,
        num_layers=1, neuron_type="rsynaptic",
        beta=0.9, threshold=1.0, gradient_slope=25.0,
        learn_beta=True, alpha=0.85, dropout=0.0,
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required for efficiency benchmark.")
    print(f"GPU: {gpu_name()}")
    print(
        f"Config: B={BATCH_SIZE}, T={HISTORY_BINS}, M={NUM_NEURONS} | "
        f"warmup={WARMUP}, timed={TIMED}"
    )

    x = torch.randn(
        BATCH_SIZE, HISTORY_BINS, NUM_NEURONS,
        device=device, dtype=torch.float32,
    )
    print(f"Input tensor: {tuple(x.shape)}  ({x.element_size() * x.nelement() / 1024**2:.1f} MB)")

    builders = [
        ("Mamba",       build_mamba),
        ("Transformer", build_transformer),
        ("LRU v2",      build_lru),
        ("LSTM",        build_lstm),
        ("SNN (1L)",    build_snn_1l),
    ]

    results = []
    for name, builder in builders:
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print('=' * 60)
        try:
            model = builder(NUM_NEURONS).to(device)
            n_params = count_params(model)
            print(f"  Params: {n_params:,}")
            result = benchmark(model, x, name)
            print(
                f"  Latency:    {result['latency_ms']:>7.3f} ms/batch "
                f"(B={BATCH_SIZE}, so {result['latency_ms']/BATCH_SIZE*1000:.3f} us/sample)"
            )
            print(f"  Peak VRAM:  {result['peak_vram_mb']:>7.1f} MB")
            print(f"  Throughput: {result['throughput_samples_per_s']:>7,.0f} samples/s")
            results.append(result)
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            results.append({
                "name": name, "error": f"{type(e).__name__}: {e}",
            })

    # Summary
    print(f"\n{'=' * 80}")
    print(f"  EFFICIENCY SUMMARY (RTX 3090, B={BATCH_SIZE}, T={HISTORY_BINS}, "
          f"M={NUM_NEURONS})")
    print('=' * 80)
    print(f"  {'Model':<14} {'Params':>10} {'Latency(ms)':>12} {'VRAM(MB)':>10} {'Throughput':>12}")
    print("  " + "-" * 60)
    for r in results:
        if "error" in r:
            print(f"  {r['name']:<14} FAILED: {r['error']}")
            continue
        print(
            f"  {r['name']:<14} {r['n_params']:>10,} "
            f"{r['latency_ms']:>12.3f} {r['peak_vram_mb']:>10.1f} "
            f"{r['throughput_samples_per_s']:>12,.0f}"
        )

    # Upload
    out = {
        "gpu": gpu_name(),
        "batch_size": BATCH_SIZE,
        "history_bins": HISTORY_BINS,
        "num_neurons": NUM_NEURONS,
        "warmup": WARMUP,
        "timed_iterations": TIMED,
        "results": results,
    }
    s3.put_object(
        Bucket=BUCKET, Key=S3_OUTPUT_KEY,
        Body=json.dumps(out, indent=2).encode(),
    )
    print(f"\nUploaded to s3://{BUCKET}/{S3_OUTPUT_KEY}")


if __name__ == "__main__":
    main()
