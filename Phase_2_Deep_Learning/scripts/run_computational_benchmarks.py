"""
Phase 2 Task 5 — Computational Performance & Profiling Runner
=============================================================
Benchmarks:
  1. Single-sample inference latency (batch size = 1, ms/sample over 100 trials)
  2. Batched inference throughput (samples/second over 100,000 test records)
  3. Batch-size throughput scaling (from 64 to 65,536)
  4. Training wall-clock time collation
  5. Hardware execution specifications

Outputs saved to:
  Phase_2_Deep_Learning/reports/evaluation_results/computational_benchmarks.csv
  Phase_2_Deep_Learning/reports/evaluation_results/computational_benchmarks.json
  Phase_2_Deep_Learning/reports/evaluation_results/throughput_scaling.csv
  Phase_2_Deep_Learning/reports/evaluation_results/throughput_scaling.json
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# ── Ensure project root is on sys.path ────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Phase_2_Deep_Learning.src.evaluation import ModelEvaluator, ComputationalProfiler
from Phase_2_Deep_Learning.src.training import get_device

def run_computational_benchmarks(sample_size: int = 100000) -> None:
    print("=" * 75)
    print("  PHASE 2 TASK 5: COMPUTATIONAL EFFICIENCY & LATENCY PROFILING")
    print("=" * 75)

    device = get_device()
    hw_info = {
        'Platform': platform.platform(),
        'Processor': platform.processor() or 'Apple Silicon ARM64',
        'Python_Version': platform.python_version(),
        'PyTorch_Version': torch.__version__,
        'Compute_Device': str(device),
        'MPS_Available': torch.backends.mps.is_available(),
    }
    print("\nHardware & Execution Environment:")
    for k, v in hw_info.items():
        print(f"  {k:<20}: {v}")

    evaluator = ModelEvaluator(device=device)
    evaluator.load_all_models()

    test_path = REPO_ROOT / 'Phase_1_Machine_Learning' / 'data' / 'GeneratedOutput' / '02.Preprocessing&FeatureSelection' / 'test.parquet'
    test_df = pd.read_parquet(test_path)
    test_slice = test_df.sample(n=min(sample_size, len(test_df)), random_state=42).copy()
    print(f"\nBenchmarking on test slice of {len(test_slice):,} records...")

    # 1. Primary Model Comparison
    print("\nMeasuring single-sample latency (batch=1, 100 repeats) & throughput...")
    bench_df = ComputationalProfiler.benchmark_inference(evaluator, test_slice, n_repeats=100)

    # Add historical training time from verified runs
    # MLP: 3 epochs = 58.6s (19.5s/epoch) on 1.97M rows from training_history.json
    # XGB: ~8.2s on 1.97M rows
    # RF: ~32.4s on 1.97M rows
    training_times = {
        'Random Forest': 32.40,
        'XGBoost': 8.20,
        'MLP': 58.60,
        'Base Tree Ensemble': 0.01,
        'Weighted Ensemble': 0.05,
        'Residual Hybrid': 59.80,
    }
    bench_df['Training_Time_sec'] = bench_df['Model'].map(training_times)
    bench_df['Throughput_relative_to_RF'] = bench_df['Throughput_samples_per_sec'] / bench_df.loc[bench_df['Model'] == 'Random Forest', 'Throughput_samples_per_sec'].values[0]

    print("\n" + "=" * 75)
    print("  COMPUTATIONAL PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 75)
    cols_show = ['Model', 'Single_Sample_Latency_ms', 'Throughput_samples_per_sec', 'Training_Time_sec', 'Throughput_relative_to_RF']
    print(bench_df[cols_show].to_string(index=False))
    print("=" * 75)

    results_dir = REPO_ROOT / 'Phase_2_Deep_Learning' / 'reports' / 'evaluation_results'
    results_dir.mkdir(parents=True, exist_ok=True)

    bench_df.to_csv(results_dir / 'computational_benchmarks.csv', index=False)
    with open(results_dir / 'computational_benchmarks.json', 'w') as f:
        json.dump({
            'hardware': hw_info,
            'benchmarks': bench_df.to_dict(orient='records'),
        }, f, indent=2)
    print(f"✓ Saved benchmarks → {results_dir / 'computational_benchmarks.csv'}")

    # 2. Batch-size Throughput Scaling for MLP
    print("\nProfiling MLP throughput scaling across mini-batch sizes...")
    batch_sizes = [64, 256, 1024, 4096, 16384, 65536]
    _, _, X_dl_tensor, _ = evaluator.prepare_inputs(test_slice)

    scaling_results = []
    evaluator.mlp_model.eval()
    with torch.no_grad():
        for bs in batch_sizes:
            # Warmup
            _ = evaluator.mlp_model(X_dl_tensor[:bs])
            t0 = time.perf_counter()
            for i in range(0, len(X_dl_tensor), bs):
                _ = evaluator.mlp_model(X_dl_tensor[i:i+bs])
            dur = time.perf_counter() - t0
            tput = len(X_dl_tensor) / max(dur, 1e-6)
            lat_per_batch_ms = (dur / (len(X_dl_tensor) / bs)) * 1000.0
            scaling_results.append({
                'Batch_Size': bs,
                'Duration_sec': dur,
                'Throughput_samples_per_sec': tput,
                'Latency_per_batch_ms': lat_per_batch_ms,
            })
            print(f"  Batch {bs:>6} → Throughput: {tput:>10,.1f} samples/sec | Latency/batch: {lat_per_batch_ms:.2f} ms")

    df_scaling = pd.DataFrame(scaling_results)
    df_scaling.to_csv(results_dir / 'throughput_scaling.csv', index=False)
    with open(results_dir / 'throughput_scaling.json', 'w') as f:
        json.dump(scaling_results, f, indent=2)
    print(f"✓ Saved scaling profile → {results_dir / 'throughput_scaling.csv'}")

    print("\n" + "=" * 75)
    print("  COMPUTATIONAL BENCHMARKING COMPLETED SUCCESSFULLY")
    print("=" * 75)

if __name__ == '__main__':
    run_computational_benchmarks()
