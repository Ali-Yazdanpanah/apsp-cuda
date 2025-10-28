#!/usr/bin/env python3
"""Visualise APSP benchmark results."""

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt

LOG = logging.getLogger("plot-benchmarks")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("reports/data/benchmark_runs.csv"),
        help="CSV file produced by scripts/benchmark.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/figures/runtime_comparison.png"),
        help="Destination image path for the runtime plot",
    )
    parser.add_argument(
        "--title",
        default="APSP Runtime Comparison",
        help="Plot title",
    )
    parser.add_argument(
        "--logy",
        action="store_true",
        help="Use logarithmic scale on the Y axis",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def load_records(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    with path.open() as fh:
        reader = csv.DictReader(fh)
        rows = []
        for row in reader:
            row = dict(row)
            row["size"] = int(row["size"])
            row["time_ms"] = float(row["time_ms"])
            rows.append(row)
    LOG.info("Loaded %d samples from %s", len(rows), path)
    return rows


def aggregate(records: Iterable[Dict[str, object]]) -> Dict[str, List[Tuple[int, float, float, float]]]:
    groups: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for row in records:
        label = row.get("label") or row.get("algorithm")
        groups[(label, int(row["size"]))].append(float(row["time_ms"]))

    aggregated: Dict[str, List[Tuple[int, float, float, float]]] = defaultdict(list)
    for (label, size), values in sorted(groups.items()):
        avg = sum(values) / len(values)
        aggregated[label].append((size, avg, min(values), max(values)))
    return aggregated


def plot(summary: Dict[str, List[Tuple[int, float, float, float]]], output: Path, *, title: str, logy: bool) -> None:
    if not summary:
        raise ValueError("No data available to plot")

    plt.figure(figsize=(10, 6))

    for label, samples in summary.items():
        sizes = [size for size, *_ in samples]
        means = [avg for _, avg, _, _ in samples]
        mins = [low for _, _, low, _ in samples]
        maxs = [high for _, _, _, high in samples]

        plt.plot(sizes, means, marker="o", label=label)
        plt.fill_between(sizes, mins, maxs, alpha=0.1)

    plt.xlabel("Vertices")
    plt.ylabel("Runtime (ms)")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    if logy:
        plt.yscale("log")
    plt.legend()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output)
    LOG.info("Saved plot to %s", output)


def main(argv: Iterable[str]) -> int:
    options = parse_args(argv)
    configure_logging(options.verbose)

    try:
        records = load_records(options.input)
    except FileNotFoundError as exc:
        LOG.error(str(exc))
        return 1

    summary = aggregate(records)
    try:
        plot(summary, options.output, title=options.title, logy=options.logy)
    except ValueError as exc:
        LOG.error(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv[1:]))
