#!/usr/bin/env python3
"""
Benchmark runners for APSP implementations.

This script executes the CPU and CUDA binaries with a configurable matrix size
suite, captures their structured stdout, and stores aggregated results to CSV.

The binaries print a single comma-separated metrics line when invoked with
``--quiet``. Example:
    floyd-warshall,size=512,iteration=0,time_ms=123.456789

The script parses these lines for all requested sizes and iterations and writes
them to ``reports/data/benchmark_runs.csv`` by default. A companion summary
file (averages per algorithm/size) is also produced.

Usage:
    python3 scripts/benchmark.py --sizes 256 512 1024 --iterations 3
"""

import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BIN_DIR = ROOT_DIR / "bin"
DEFAULT_REPORT_DIR = ROOT_DIR / "reports" / "data"

LOG = logging.getLogger("benchmark")

# Regular expression to capture key=value pairs from algorithm output.
PAIR_PATTERN = re.compile(r"(?P<key>[^=,]+)=(?P<value>[^,]+)")

ALGORITHMS = [
    {
        "label": "Floyd CPU",
        "binary": "floyd_cpu",
        "density": 0.30,
        "extra_args": [],
    },
    {
        "label": "Floyd GPU",
        "binary": "floyd_gpu",
        "density": 0.30,
        "extra_args": [],
        "optional": True,
    },
    {
        "label": "Dijkstra CPU",
        "binary": "dijkstra_cpu",
        "density": 0.15,
        "extra_args": [],
    },
    {
        "label": "Dijkstra GPU",
        "binary": "dijkstra_gpu",
        "density": 0.15,
        "extra_args": [],
        "optional": True,
    },
]


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bin-dir",
        type=Path,
        default=DEFAULT_BIN_DIR,
        help="Directory containing compiled binaries (default: %(default)s)",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[256, 512, 1024],
        help="Graph sizes (number of vertices) to benchmark",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations to execute per benchmark run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=777,
        help="Seed forwarded to binaries for reproducibility",
    )
    parser.add_argument(
        "--max-weight",
        type=int,
        default=100,
        help="Maximum edge weight used during random graph generation",
    )
    parser.add_argument(
        "--infinity",
        type=int,
        default=999999,
        help="Infinity sentinel value forwarded to binaries",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_DIR / "benchmark_runs.csv",
        help="CSV file to store raw benchmark samples",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_REPORT_DIR / "benchmark_summary.csv",
        help="CSV file to store aggregated benchmark summary",
    )
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="Skip GPU binaries even if they exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(message)s",
    )


def parse_metrics_line(line: str) -> Dict[str, str]:
    tokens = line.strip().split(",")
    if not tokens:
        raise ValueError(f"Unexpected output line: {line!r}")
    result: Dict[str, str] = {"algorithm": tokens[0]}
    for match in PAIR_PATTERN.finditer(line):
        result[match.group("key")] = match.group("value")
    return result


def run_command(cmd: List[str], *, dry_run: bool = False) -> Tuple[int, str, str]:
    LOG.debug("Running: %s", " ".join(cmd))
    if dry_run:
        return 0, "", ""
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def extract_latency(record: Dict[str, str]) -> float:
    for key in ("time_ms", "gpu_time_ms", "wall_time_ms"):
        if key in record:
            return float(record[key])
    raise KeyError(f"No timing metric found in record: {json.dumps(record, indent=2)}")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def benchmark_algorithm(
    binary_path: Path,
    algorithm_cfg: Dict[str, object],
    sizes: Iterable[int],
    options: argparse.Namespace,
    records: List[Dict[str, object]],
) -> None:
    density = algorithm_cfg.get("density", 0.25)
    extra_args = list(algorithm_cfg.get("extra_args", []))

    for size in sizes:
        cmd = [
            str(binary_path),
            "--size",
            str(size),
            "--density",
            f"{density:.6f}",
            "--iterations",
            str(options.iterations),
            "--seed",
            str(options.seed),
            "--max-weight",
            str(options.max_weight),
            "--infinity",
            str(options.infinity),
            "--quiet",
        ]
        cmd.extend(extra_args)

        code, stdout, stderr = run_command(cmd, dry_run=options.dry_run)
        if code != 0:
            LOG.warning(
                "Command failed (%s): %s\nstderr: %s",
                code,
                " ".join(cmd),
                stderr.strip(),
            )
            continue
        if options.dry_run:
            continue

        lines = [line for line in stdout.splitlines() if line.strip()]
        if not lines:
            LOG.warning("No metrics captured from %s", " ".join(cmd))
            continue

        for line in lines:
            parsed = parse_metrics_line(line)
            try:
                duration_ms = extract_latency(parsed)
            except KeyError as exc:
                LOG.warning("Skipping line due to missing metric: %s (%s)", line, exc)
                continue

            record = {
                "label": algorithm_cfg["label"],
                "binary": binary_path.name,
                "size": int(parsed.get("size", size)),
                "density": float(density),
                "iteration": int(parsed.get("iteration", 0)),
                "time_ms": duration_ms,
            }

            # Persist any other key-values captured in the output.
            for key, value in parsed.items():
                if key in {"algorithm", "size", "iteration", "time_ms", "gpu_time_ms", "wall_time_ms"}:
                    continue
                record[key] = value
            records.append(record)


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        LOG.warning("No rows to write to %s", path)
        return

    ensure_parent(path)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    LOG.info("Wrote %d rows to %s", len(rows), path)


def aggregate_records(records: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for row in records:
        groups[(row["label"], int(row["size"]))].append(float(row["time_ms"]))

    summary: List[Dict[str, object]] = []
    for (label, size), values in sorted(groups.items()):
        summary.append(
            {
                "label": label,
                "size": size,
                "samples": len(values),
                "time_ms_avg": sum(values) / len(values),
                "time_ms_min": min(values),
                "time_ms_max": max(values),
            }
        )
    return summary


def main(argv: Iterable[str]) -> int:
    options = parse_args(argv)
    configure_logging(options.verbose)

    if not options.bin_dir.exists():
        LOG.error("Binary directory %s does not exist", options.bin_dir)
        return 1

    records: List[Dict[str, object]] = []
    for algo in ALGORITHMS:
        binary_path = options.bin_dir / algo["binary"]
        if not binary_path.exists():
            if algo.get("optional") or options.skip_gpu:
                LOG.info("Skipping %s (binary not found)", algo["label"])
                continue
            LOG.warning("Required binary missing: %s", binary_path)
            continue
        if options.skip_gpu and "GPU" in algo["label"]:
            LOG.info("Skipping %s per --skip-gpu flag", algo["label"])
            continue
        LOG.info("Running %s", algo["label"])
        benchmark_algorithm(binary_path, algo, options.sizes, options, records)

    if options.dry_run:
        return 0

    write_csv(options.output, records)
    summary = aggregate_records(records)
    write_csv(options.summary_output, summary)

    return 0 if records else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
