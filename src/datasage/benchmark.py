"""Reproducible synthetic issue benchmark for DataSage.

Measures candidate issue retrieval, not causal correctness or generalization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import pandas as pd

from datasage.workflow import build_workflow


def benchmark_cases() -> list[tuple[str, pd.DataFrame, str | None, set[tuple[str, str]]]]:
    """Fixed, explicitly labeled cases. Keys are (category, column)."""
    base = pd.DataFrame({"value": [float(i % 17) for i in range(100)],
                         "group": ["a", "b"] * 50})
    missing = base.copy()
    missing.loc[0:24, "value"] = float("nan")
    duplicates = base.copy()
    duplicates.loc[50:59] = duplicates.loc[0:9].to_numpy()
    outliers = pd.DataFrame({"value": list(range(99)) + [10000.0]})
    imbalance = pd.DataFrame({"target": ["majority"] * 90 + ["minority"] * 10})
    proxy = pd.DataFrame({"target": [i % 2 for i in range(100)],
                          "outcome_copy": [i % 2 for i in range(100)]})
    clean = pd.DataFrame({"group": ["a", "b"] * 50, "value": [i % 17 for i in range(100)]})
    return [
        ("missing", missing, None, {("missing_data", "value")}),
        ("duplicates", duplicates, None, {("duplicate_rows", "")}),
        ("outlier", outliers, None, {("statistical_outlier", "value")}),
        ("imbalance", imbalance, "target", {("class_imbalance", "target")}),
        ("target_proxy", proxy, "target", {("target_proxy_candidate", "outcome_copy")}),
        ("clean", clean, None, set()),
    ]


def observed_keys(report: dict, categories: set[str]) -> set[tuple[str, str]]:
    """Only evidence-verified findings in benchmarked categories count as detections."""
    return {
        (item["category"], item.get("column") or "")
        for item in report["findings"]["verified"]
        if item["category"] in categories
    }


def score_benchmark() -> dict:
    cases = benchmark_cases()
    categories = {category for _, _, _, truth in cases for category, _ in truth}
    rows: list[dict] = []
    totals = {"tp": 0, "fp": 0, "fn": 0}
    graph = build_workflow()
    with tempfile.TemporaryDirectory(prefix="datasage-benchmark-") as folder:
        for name, frame, target, truth in cases:
            path = Path(folder) / (name + ".csv")
            frame.to_csv(path, index=False)
            request = {"dataset_path": str(path)}
            if target is not None:
                request["target_column"] = target
            report = graph.invoke(request)["final_report"]
            predicted = observed_keys(report, categories)
            tp, fp, fn = len(truth & predicted), len(predicted - truth), len(truth - predicted)
            for key, val in (("tp", tp), ("fp", fp), ("fn", fn)):
                totals[key] += val
            rows.append({
                "case": name,
                "truth": sorted([list(x) for x in truth]),
                "predicted": sorted([list(x) for x in predicted]),
                "tp": tp, "fp": fp, "fn": fn,
                "unverified_count": report["summary"]["not_verified"],
                "rejected_count": report["summary"]["rejected"],
            })
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    return {
        "benchmark_version": "0.1",
        "scope": "Six small deterministic synthetic scenarios; evidence-verified retrieval only",
        "totals": totals,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "cases": rows,
        "limitations": [
            "These scenarios were designed around known current detectors, so scores cannot demonstrate generalization.",
            "The clean case is clean only for benchmarked labels, not every possible data issue.",
            "A verified calculation is not proof of real-world defect or causality.",
            "No independent competing system or statistical confidence interval is evaluated.",
            "Only five categories enter TP/FP/FN; unverified findings are excluded.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DataSage synthetic benchmark")
    parser.add_argument("--output", type=Path, default=None, help="Save JSON summary to a file")
    args = parser.parse_args()
    result = score_benchmark()
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
