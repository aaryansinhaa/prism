#!/usr/bin/env python3
"""Generate polished benchmark plots from JSON reports.

Reads benchmark JSON files from `docs/benchmarks/` and writes PNG figures to
`docs/benchmarks/figures/`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class Report:
    name: str
    path: Path
    payload: dict[str, Any]


def _load_reports(input_dir: Path) -> list[Report]:
    reports: list[Report] = []
    for path in sorted(input_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        reports.append(Report(name=path.stem, path=path, payload=payload))
    return reports


def _shorten(name: str) -> str:
    name = name.replace("runtime.adapters.", "")
    name = name.replace("app.core.", "")
    name = name.replace("app.batching.", "")
    name = name.replace("app.routing.", "")
    name = name.replace(".predict_model", "")
    return name


def _section_rows(payload: dict[str, Any], section: str) -> list[dict[str, Any]]:
    raw = payload.get(section, [])
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _plot_module_metric_grid(report: Report, metric_key: str, title: str, out_file: Path) -> None:
    sections = ["runtime", "core", "routes"]
    colors = {
        "runtime": "#4C78A8",
        "core": "#59A14F",
        "routes": "#F28E2B",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    for axis, section in zip(axes, sections):
        rows = _section_rows(report.payload, section)
        labels = [_shorten(str(row.get("name", "unknown"))) for row in rows]
        values = [float(row.get(metric_key, 0.0) or 0.0) for row in rows]

        order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
        labels = [labels[i] for i in order]
        values = [values[i] for i in order]

        axis.barh(labels, values, color=colors[section], alpha=0.9)
        axis.invert_yaxis()
        axis.set_title(section.upper(), fontsize=12, fontweight="bold")
        axis.grid(axis="x", linestyle="--", alpha=0.25)

        for index, value in enumerate(values):
            axis.text(value, index, f" {value:.2f}", va="center", fontsize=9)

    fig.suptitle(f"{title}\nSource: {report.name}", fontsize=14, fontweight="bold")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_batching_comparison(report: Report, out_file: Path) -> None:
    comparison = report.payload.get("batching_comparison", {})
    if not isinstance(comparison, dict):
        return

    baseline = comparison.get("baseline", {})
    batched = comparison.get("batched", {})
    improvement = comparison.get("improvement", {})

    if not isinstance(baseline, dict) or not isinstance(batched, dict):
        return

    metrics = [
        ("avg_latency_ms", "Avg Latency (ms)", True),
        ("p95_latency_ms", "P95 Latency (ms)", True),
        ("throughput_rps", "Throughput (rps)", False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    for axis, (key, label, lower_is_better) in zip(axes, metrics):
        before = float(baseline.get(key, 0.0) or 0.0)
        after = float(batched.get(key, 0.0) or 0.0)
        axis.bar(["Baseline", "Batched"], [before, after], color=["#9C755F", "#2E8B57"])
        axis.set_title(label, fontsize=11, fontweight="bold")
        axis.grid(axis="y", linestyle="--", alpha=0.3)

        if before > 0:
            delta_pct = ((after - before) / before) * 100.0
            if lower_is_better:
                score = -delta_pct
                note = f"improvement: {score:.2f}%"
            else:
                score = delta_pct
                note = f"improvement: {score:.2f}%"
            axis.text(0.5, max(before, after) * 1.05, note, ha="center", fontsize=9)

    latency_imp = float(improvement.get("avg_latency_reduction_pct", 0.0) or 0.0)
    thr_imp = float(improvement.get("throughput_increase_pct", 0.0) or 0.0)
    fig.suptitle(
        "Batching Impact (Inference Route)"
        f"\n{report.name} | latency reduction={latency_imp:.2f}% | throughput increase={thr_imp:.2f}%",
        fontsize=13,
        fontweight="bold",
    )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_improvement_across_reports(reports: list[Report], out_file: Path) -> None:
    labels: list[str] = []
    latency_reduction: list[float] = []
    throughput_increase: list[float] = []

    for report in reports:
        comparison = report.payload.get("batching_comparison", {})
        if not isinstance(comparison, dict):
            continue
        improvement = comparison.get("improvement", {})
        if not isinstance(improvement, dict):
            continue

        labels.append(report.name)
        latency_reduction.append(float(improvement.get("avg_latency_reduction_pct", 0.0) or 0.0))
        throughput_increase.append(float(improvement.get("throughput_increase_pct", 0.0) or 0.0))

    if not labels:
        return

    x = range(len(labels))
    width = 0.38

    fig, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    axis.bar([i - width / 2 for i in x], latency_reduction, width=width, label="Latency Reduction %", color="#4C78A8")
    axis.bar([i + width / 2 for i in x], throughput_increase, width=width, label="Throughput Increase %", color="#59A14F")

    axis.axhline(0, color="black", linewidth=1)
    axis.set_xticks(list(x), labels, rotation=10)
    axis.set_ylabel("Percent (%)")
    axis.set_title("Batching Improvement Across Benchmark Profiles", fontsize=13, fontweight="bold")
    axis.legend()
    axis.grid(axis="y", linestyle="--", alpha=0.3)

    for i, value in enumerate(latency_reduction):
        axis.text(i - width / 2, value, f" {value:.1f}", va="bottom" if value >= 0 else "top", fontsize=8)
    for i, value in enumerate(throughput_increase):
        axis.text(i + width / 2, value, f" {value:.1f}", va="bottom" if value >= 0 else "top", fontsize=8)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("docs/benchmarks"),
        help="Directory containing benchmark JSON reports.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/benchmarks/figures"),
        help="Directory where PNG figures are written.",
    )
    parser.add_argument(
        "--primary-report",
        type=str,
        default="latest",
        help="Report stem to use for per-module charts (default: latest).",
    )
    args = parser.parse_args()

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "figure.titlesize": 14,
        }
    )

    reports = _load_reports(args.input_dir)
    if not reports:
        raise SystemExit(f"No benchmark JSON files found in: {args.input_dir}")

    primary = next((item for item in reports if item.name == args.primary_report), reports[0])

    _plot_module_metric_grid(
        report=primary,
        metric_key="avg_ms",
        title="Average Latency by Module",
        out_file=args.output_dir / "module_avg_latency.png",
    )
    _plot_module_metric_grid(
        report=primary,
        metric_key="throughput_rps",
        title="Throughput by Module",
        out_file=args.output_dir / "module_throughput.png",
    )
    _plot_batching_comparison(primary, args.output_dir / f"batching_comparison_{primary.name}.png")
    _plot_improvement_across_reports(reports, args.output_dir / "batching_improvement_across_reports.png")

    print(f"Generated benchmark figures in: {args.output_dir}")


if __name__ == "__main__":
    main()
