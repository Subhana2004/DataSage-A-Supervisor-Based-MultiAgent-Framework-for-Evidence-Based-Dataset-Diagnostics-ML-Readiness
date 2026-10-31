"""External dataset evaluation harness, with explicitly non-ground-truth baselines.

UCI datasets are fetched only when the evaluator explicitly runs --fetch.
The pandas baseline evaluates descriptive counts, not leakage causality.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile

import pandas as pd
from datasage.workflow import build_workflow


def pandas_baseline(frame: pd.DataFrame) -> set[tuple[str, str]]:
    """Conventional descriptive EDA: exact missingness, duplicates, constants."""
    result = set()
    for col in frame.columns:
        if frame[col].isna().any():
            result.add(("missing_data", str(col)))
        if len(frame[col].dropna()) > 0 and frame[col].nunique(dropna=True) == 1:
            result.add(("constant_feature", str(col)))
    if len(frame.columns) and frame.duplicated().any():
        result.add(("duplicate_rows", ""))
    return result


def compare_frame(frame: pd.DataFrame, target: str | None = None) -> dict:
    """Compare descriptions, not truth labels or accuracy scores."""
    with tempfile.TemporaryDirectory(prefix="datasage-eval-") as folder:
        path = Path(folder) / "input.csv"
        frame.to_csv(path, index=False)
        request = {"dataset_path": str(path)}
        if target:
            request["target_column"] = target
        report = build_workflow().invoke(request)["final_report"]
    categories = {"missing_data", "constant_feature", "duplicate_rows"}
    verified = {
        (item["category"], item.get("column") or "")
        for item in report["findings"]["verified"]
        if item["category"] in categories
    }
    baseline = pandas_baseline(frame)
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "baseline": sorted([list(x) for x in baseline]),
        "datasage_verified": sorted([list(x) for x in verified]),
        "agreement": sorted([list(x) for x in baseline & verified]),
        "baseline_only": sorted([list(x) for x in baseline - verified]),
        "datasage_only": sorted([list(x) for x in verified - baseline]),
        "other_verified_categories": sorted({
            item["category"] for item in report["findings"]["verified"]
            if item["category"] not in categories
        }),
        "unverified_count": report["summary"]["not_verified"],
        "rejected_count": report["summary"]["rejected"],
    }


def fetch_uci_frames() -> list[tuple[str, pd.DataFrame, str]]:
    """UCI machine-learning repository IDs, via its maintained client.

    This is a network operation; never run it as part of unit tests.
    """
    from ucimlrepo import fetch_ucirepo
    output = []
    for dataset_id, name, target in [(2, "adult", "income"), (186, "wine_quality", "quality")]:
        record = fetch_ucirepo(id=dataset_id)
        x = record.data.features.reset_index(drop=True)
        y = record.data.targets.reset_index(drop=True)
        if len(y.columns) != 1:
            raise ValueError("Expected a single target")
        frame = pd.concat([x, y.rename(columns={y.columns[0]: target})], axis=1)
        if frame.columns.duplicated().any():
            raise ValueError("Duplicate names in fetched data")
        output.append((name, frame, target))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true", help="Explicitly download UCI data")
    parser.add_argument("--output", type=Path, default=Path("results/external-evaluation.json"))
    args = parser.parse_args()
    if not args.fetch:
        parser.error("Use --fetch to explicitly download external datasets")
    rows = []
    for name, frame, target in fetch_uci_frames():
        result = compare_frame(frame, target)
        result["dataset"] = name
        rows.append(result)
    output = {
        "evaluation": "external_descriptive_agreement_v0.1",
        "datasets": rows,
        "ground_truth_available": False,
        "accuracy_claim_allowed": False,
        "limitations": [
            "Dataset source documentation is not a set of independently annotated anomaly labels.",
            "Baseline shares pandas primitives with DataSage; agreement is not independent correctness.",
            "Anomalies, class imbalance and target-proxy warnings require task-specific human review.",
            "Source representations and missing markers must be inspected before interpreting disagreement.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(output["datasets"], indent=2))


if __name__ == "__main__":
    main()
