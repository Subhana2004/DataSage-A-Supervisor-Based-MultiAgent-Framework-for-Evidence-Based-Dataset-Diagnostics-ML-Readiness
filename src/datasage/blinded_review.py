"""Prepare blinded diagnostic review cases and score independent annotations.

Does not invent ground truth; two human reviewers must provide labels.
"""
from __future__ import annotations
import hashlib
import json
from typing import Any

LABELS = {"actionable", "not_actionable", "uncertain"}


def prepare_blinded_review(datasage_report: dict, baseline_keys: set[tuple[str, str]],
                           dataset_id: str) -> tuple[list[dict], dict[str, dict]]:
    """Return (review_cases, sealed_predictions); do not show sealed map to reviewers."""
    findings = {}
    for status, entries in datasage_report["findings"].items():
        for item in entries:
            key = (item["category"], item.get("column") or "")
            findings[key] = {
                "category": key[0], "column": key[1],
                "description": item.get("description", ""),
                "evidence_summary": item.get("evidence", {}).get("metrics", {}),
                "validation_status": status,
            }
    candidates = sorted(set(findings) | baseline_keys)
    cases, sealed = [], {}
    for category, column in candidates:
        case_id = hashlib.sha256(json.dumps([dataset_id, category, column]).encode()).hexdigest()[:16]
        item = findings.get((category, column), {})
        cases.append({
            "case_id": case_id, "dataset_id": dataset_id,
            "category": category, "column": column,
            "observation": item.get("description", "Detected by a descriptive baseline"),
            "evidence_summary": item.get("evidence_summary", {}),
            "review_question": "Is a concrete data-quality action justified under the documented task and data-collection context?",
        })
        sealed[case_id] = {
            "datasage": bool((category, column) in findings and item.get("validation_status") == "verified"),
            "baseline": (category, column) in baseline_keys,
        }
    return cases, sealed


def adjudicate_reviews(first: dict[str, str], second: dict[str, str],
                       case_ids: set[str]) -> dict[str, str]:
    """Include only cases where both reviewers agree on an actionable/nonactionable label."""
    if not set(first).issubset(case_ids) or not set(second).issubset(case_ids):
        raise ValueError("Reviewer labels contain unknown case IDs")
    result = {}
    for case_id in case_ids:
        a, b = first.get(case_id), second.get(case_id)
        if a is not None and a not in LABELS or b is not None and b not in LABELS:
            raise ValueError("Invalid reviewer label")
        result[case_id] = a if a == b and a in {"actionable", "not_actionable"} else "unresolved"
    return result


def score_review(predictions: dict[str, dict], consensus: dict[str, str], system: str) -> dict[str, Any]:
    """Evaluation on resolved cases only, with explicit exclusion counts."""
    if system not in {"datasage", "baseline"}:
        raise ValueError("Unknown system")
    tp = fp = fn = tn = unresolved = 0
    for case_id, methods in predictions.items():
        label = consensus.get(case_id, "unresolved")
        if label not in {"actionable", "not_actionable"}:
            unresolved += 1
            continue
        predicted = bool(methods[system])
        if predicted and label == "actionable":
            tp += 1
        elif predicted:
            fp += 1
        elif label == "actionable":
            fn += 1
        else:
            tn += 1
    return {
        "system": system, "resolved_cases": tp + fp + fn + tn,
        "unresolved_cases": unresolved,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "limitations": "Only concordant human labels; no claim about causality or performance outside this reviewed sample.",
    }
