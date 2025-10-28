# APSP Acceleration Suite

Modernised all-pairs shortest path (APSP) playground featuring clean C/CUDA
implementations, reproducible benchmarking, and reporting utilities tailored
for portfolio presentation.

## Highlights

- **Algorithms**
  - `floyd_cpu`: Optimised Floyd–Warshall using configurable random graphs and
    structured timing output.
  - `dijkstra_cpu`: Rewritten multi-source Dijkstra with verification option
    against a Floyd baseline.
  - `floyd_gpu`: CUDA kernel with host-side verification, per-block controls,
    and CSV logging.
  - `dijkstra_gpu`: Hybrid CPU/GPU relaxation pipeline that offloads edge
    relaxation while keeping deterministic host control.
- **Tooling**: Python benchmarking harness (`scripts/benchmark.py`) and chart
  generator (`scripts/plot_benchmarks.py`) to capture results and visualise
  scaling behaviour.
- **Artifacts**: Ready-to-share performance report (`reports/`) with data and
  figures produced by the tooling.

## Requirements

- GCC with C11 support
- Python 3.9+ with `matplotlib` (install via `pip install matplotlib`)
- Optional for GPU targets: NVIDIA CUDA Toolkit (`nvcc`) and a CUDA-capable device

## Quick Start

```bash
# Clone the repository then inside the project root:
make            # Builds CPU + GPU binaries into ./bin

# or build subsets
make cpu        # Only CPU binaries
make gpu        # Only CUDA binaries (requires nvcc)
```

All binaries support `--help` for full CLI details.

Example run:

```bash
bin/floyd_cpu --size 512 --density 0.25 --iterations 5 --seed 42 --csv runs.csv
bin/dijkstra_cpu --size 512 --density 0.15 --verify --quiet
```

## Benchmark & Reporting Workflow

1. Execute the benchmark suite (skips GPU binaries gracefully when unsupported):

   ```bash
   python3 scripts/benchmark.py --sizes 128 256 512 --iterations 2 --seed 42
   ```

   Raw results land in `reports/data/benchmark_runs.csv`; a companion summary is
   stored at `reports/data/benchmark_summary.csv`.

2. Generate runtime charts:

   ```bash
   python3 scripts/plot_benchmarks.py
   ```

   The default figure is written to `reports/figures/runtime_comparison.png`.

3. Review the ready-made write-up at `reports/performance_report.md`, which
   references both the data table and figure.

## Project Layout

```
├── bin/                     # Compiled executables (generated)
├── include/                 # Shared headers
├── src/                     # Shared C utilities
├── reports/                 # Benchmark data, figures, and Markdown report
├── scripts/                 # Benchmark + plotting utilities
├── floyd.c / Dijkstra.c     # CPU implementations
├── floyd.cu / Dijkstra.cu   # CUDA implementations
├── Makefile
└── README.md
```

## Notes & Future Work

- GPU binaries automatically log a clear warning if CUDA memory cannot be
  allocated (e.g., when running inside a sandbox without device access).
- The hybrid CUDA Dijkstra currently relies on host-side vertex selection; a
  fully device-side priority queue is a natural next step.
- TODO:
  1. Add directed graph visualisation using NetworkX for small instances.
  2. Integrate unit tests covering graph generation edge cases.
  3. Capture energy metrics via NVIDIA Nsight on supported hardware.


