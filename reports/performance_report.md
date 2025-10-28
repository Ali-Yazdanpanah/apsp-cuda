# Performance Snapshot

This report captures a sample benchmark run of the modernised APSP suite
(`floyd_cpu` and `dijkstra_cpu`) on the current workstation. CUDA binaries were
invoked but skipped automatically because device memory is not available in the
sandbox.

## Configuration

- Graph sizes: 128, 256, 512 vertices
- Iterations: 2 per algorithm/size
- RNG seed: 42
- Edge density: 0.30 (Floyd), 0.15 (Dijkstra)
- Max weight: 100, infinity sentinel: 999999

Commands executed:

```bash
python3 scripts/benchmark.py --sizes 128 256 512 --iterations 2 --seed 42
python3 scripts/plot_benchmarks.py
```

## Runtime Summary (ms)

| Algorithm      | Size | Avg   | Min   | Max   |
|----------------|-----:|------:|------:|------:|
| Floyd CPU      |  128 |   3.80 |   3.33 |   4.28 |
| Floyd CPU      |  256 |  20.08 |  19.27 |  20.89 |
| Floyd CPU      |  512 | 164.96 | 161.41 | 168.50 |
| Dijkstra CPU   |  128 |   7.79 |   6.98 |   8.60 |
| Dijkstra CPU   |  256 |  55.19 |  54.83 |  55.55 |
| Dijkstra CPU   |  512 | 434.65 | 422.85 | 446.45 |

See `reports/data/benchmark_summary.csv` for the raw figures captured by the
benchmark harness.

## Runtime Profile

![Runtime Comparison](figures/runtime_comparison.png)

The plot visualises how the CPU implementations scale with graph size. As GPU
hardware was unavailable, GPU data is intentionally absent. When executed on a
CUDA-capable workstation, the same workflow will automatically incorporate the
GPU binaries into the CSV and plot.
