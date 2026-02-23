#!/usr/bin/env python3
"""
Generate high-quality performance plots from benchmark data.

Reads reports/data/benchmark_runs.csv and produces:
  1. Execution Time vs. N (log-log): Floyd-Warshall and Dijkstra scaling
  2. Throughput (GB/s) vs. N: GPU warm-up and saturation effects

Usage:
    python3 scripts/plot_performance.py
    python3 scripts/plot_performance.py --min-n 128 --max-n 4096

Generate benchmark data first:
    python3 scripts/benchmark.py --sizes 256 512 1024 2048 4096 --iterations 3
"""

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

LOG = logging.getLogger("plot-performance")

# Bytes transferred for throughput (approximate)
FLOYD_BYTES_PER_N = 12  # ~12*n³ bytes per run
DIJKSTRA_BYTES_PER_N = 12  # ~12*n³ bytes per run

# Color palette suitable for research papers (colorblind-friendly)
COLORS = {
    "Floyd CPU": "#0173B2",
    "Floyd GPU": "#DE8F05",
    "Dijkstra CPU": "#029E73",
    "Dijkstra GPU": "#CC78BC",
}

MARKERS = {"Floyd CPU": "s", "Floyd GPU": "o", "Dijkstra CPU": "^", "Dijkstra GPU": "D"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("reports/data/benchmark_runs.csv"),
        help="Input CSV from benchmark.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/figures"),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=256,
        help="Minimum N for plots (default: 256)",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=4096,
        help="Maximum N for plots (default: 4096)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI (default: 150)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    records = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            r = dict(row)
            r["size"] = int(r["size"])
            # Support both time_ms and gpu_time_ms / wall_time_ms
            time_key = next(
                (k for k in ("time_ms", "gpu_time_ms", "wall_time_ms") if k in r),
                "time_ms",
            )
            r["time_ms"] = float(r[time_key])
            # Optional: gbps from CSV if present
            for k in ("gbps", "effective_gbps"):
                if k in r and r[k]:
                    try:
                        r["gbps"] = float(r[k])
                    except (ValueError, TypeError):
                        pass
            records.append(r)
    LOG.info("Loaded %d records from %s", len(records), path)
    return records


def compute_throughput_gbps(record: Dict[str, Any]) -> float:
    """Compute effective GB/s from time and size."""
    if "gbps" in record and record["gbps"]:
        return float(record["gbps"])
    n = record["size"]
    time_sec = record["time_ms"] / 1000.0
    label = record.get("label", "")
    if "Floyd" in label:
        bytes_total = FLOYD_BYTES_PER_N * (n**3)
    else:
        bytes_total = DIJKSTRA_BYTES_PER_N * (n**3)
    return bytes_total / (time_sec * 1e9) if time_sec > 0 else 0.0


def apply_paper_style() -> None:
    """Apply professional style suitable for research papers."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        }
    )
    if HAS_SEABORN:
        sns.set_theme(
            style="whitegrid",
            context="paper",
            font="serif",
            rc={"axes.spines.top": False, "axes.spines.right": False},
        )


def aggregate_by_label_size(
    records: List[Dict[str, Any]], n_min: int, n_max: int
) -> Dict[str, List[Tuple[int, float, float, float]]]:
    """Aggregate: (label, size) -> [(size, mean_time, min_time, max_time), ...]"""
    groups: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for r in records:
        if not (n_min <= r["size"] <= n_max):
            continue
        label = r.get("label", r.get("binary", "unknown"))
        groups[(label, r["size"])].append(r["time_ms"])

    result: Dict[str, List[Tuple[int, float, float, float]]] = defaultdict(list)
    for (label, size), times in sorted(groups.items()):
        result[label].append((size, np.mean(times), min(times), max(times)))
    for label in result:
        result[label].sort(key=lambda x: x[0])
    return result


def aggregate_throughput(
    records: List[Dict[str, Any]], n_min: int, n_max: int, gpu_only: bool = False
) -> Dict[str, List[Tuple[int, float, float, float, List[float]]]]:
    """Aggregate throughput per (label, size). Returns (size, mean_gbps, min, max, [per_iter])."""
    groups: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for r in records:
        if not (n_min <= r["size"] <= n_max):
            continue
        label = r.get("label", r.get("binary", "unknown"))
        if gpu_only and "GPU" not in label:
            continue
        gbps = compute_throughput_gbps(r)
        groups[(label, r["size"])].append(gbps)

    result: Dict[str, List[Tuple[int, float, float, float, List[float]]]] = defaultdict(
        list
    )
    for (label, size), gbps_list in sorted(groups.items()):
        result[label].append(
            (size, np.mean(gbps_list), min(gbps_list), max(gbps_list), gbps_list)
        )
    for label in result:
        result[label].sort(key=lambda x: x[0])
    return result


def plot_execution_time(
    summary: Dict[str, List[Tuple[int, float, float, float]]],
    output_path: Path,
    dpi: int,
) -> None:
    """Execution Time vs. N on log-log scale."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for label in sorted(summary.keys(), key=lambda x: ("GPU" in x, x)):
        data = summary[label]
        sizes = np.array([x[0] for x in data])
        means = np.array([x[1] for x in data])
        mins = np.array([x[2] for x in data])
        maxs = np.array([x[3] for x in data])
        color = COLORS.get(label, None)
        marker = MARKERS.get(label, "o")
        ax.loglog(
            sizes,
            means,
            marker=marker,
            markersize=8,
            linewidth=2,
            label=label,
            color=color,
        )
        ax.fill_between(
            sizes, mins, maxs, alpha=0.15, color=color,
        )

    ax.set_xlabel(r"Problem size $N$ (vertices)")
    ax.set_ylabel("Execution time (ms)")
    ax.set_title("Execution Time vs. Problem Size")
    ax.legend(loc="upper left", frameon=True)
    all_sizes = [s[0] for d in summary.values() for s in d]
    if all_sizes:
        ax.set_xlim(left=min(all_sizes) * 0.9, right=max(all_sizes) * 1.1)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved %s", output_path)


def plot_throughput(
    summary: Dict[str, List[Tuple[int, float, float, float, List[float]]]],
    output_path: Path,
    dpi: int,
) -> None:
    """Throughput (GB/s) vs. N showing warm-up and saturation."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for label in sorted(summary.keys(), key=lambda x: ("Floyd" in x, x)):
        data = summary[label]
        sizes = np.array([x[0] for x in data])
        means = np.array([x[1] for x in data])
        mins = np.array([x[2] for x in data])
        maxs = np.array([x[3] for x in data])
        per_iter = [x[4] for x in data]
        color = COLORS.get(label, None)
        marker = MARKERS.get(label, "o")
        ax.plot(
            sizes,
            means,
            marker=marker,
            markersize=8,
            linewidth=2,
            label=label,
            color=color,
        )
        ax.fill_between(
            sizes, mins, maxs, alpha=0.2, color=color,
        )
        # Overlay individual iteration points to show warm-up spread
        all_sizes = []
        all_gbps = []
        for s, gbps_list in zip(sizes, per_iter):
            all_sizes.extend([s] * len(gbps_list))
            all_gbps.extend(gbps_list)
        if len(all_sizes) > len(sizes):
            ax.scatter(
                all_sizes,
                all_gbps,
                alpha=0.35,
                s=25,
                color=color,
                zorder=0,
                edgecolors="none",
            )

    ax.set_xlabel(r"Problem size $N$ (vertices)")
    ax.set_ylabel("Effective throughput (GB/s)")
    ax.set_title("Throughput vs. Problem Size (warm-up and saturation)")
    ax.legend(loc="best", frameon=True)
    ax.set_xscale("log")  # N from 256 to 4096
    # Linear y to show plateau as GPU saturates at large N
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved %s", output_path)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    try:
        records = load_records(args.input)
    except FileNotFoundError as e:
        LOG.error("%s", e)
        return 1

    if not records:
        LOG.error("No records loaded")
        return 1

    apply_paper_style()

    time_summary = aggregate_by_label_size(
        records, args.min_n, args.max_n
    )
    if time_summary:
        plot_execution_time(
            time_summary,
            args.output_dir / "execution_time_vs_n.png",
            args.dpi,
        )

    throughput_summary = aggregate_throughput(
        records, args.min_n, args.max_n, gpu_only=True
    )
    if not throughput_summary:
        throughput_summary = aggregate_throughput(
            records, args.min_n, args.max_n, gpu_only=False
        )
    if throughput_summary:
        plot_throughput(
            throughput_summary,
            args.output_dir / "throughput_vs_n.png",
            args.dpi,
        )

    if not time_summary and not throughput_summary:
        LOG.warning(
            "No data in range N=%d..%d; adjust --min-n/--max-n or run benchmark with larger sizes",
            args.min_n,
            args.max_n,
        )
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
