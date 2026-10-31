"""Independent deterministic checks of diagnostic evidence.

Validation confirms supported calculations, not causal interpretation.
"""
from __future__ import annotations

import math

import pandas as pd
from scipy.stats import chi2_contingency

from datasage.anomaly import outlier_summary
from datasage.distribution import distribution_findings
from datasage.models import Finding
from datasage.ml_risk import ml_risk_findings


def validate_findings(frame: pd.DataFrame, findings: list[Finding]) -> list[dict[str, str]]:
    outcomes: list[dict[str, str]] = []
    seen: set[str] = set()
    for finding in findings:
        verdict, reason = "not_verified", "No independent validator implemented for this method"
        column = finding.column
        evidence = finding.evidence
        metrics = evidence.metrics
        if finding.finding_id in seen:
            verdict, reason = "rejected", "Duplicate finding ID"
        elif evidence.method in ("identifier_name_uniqueness", "paired_target_agreement"):
            if column not in frame.columns or not frame.columns.is_unique:
                verdict, reason = "rejected", "Feature column missing or ambiguous"
            else:
                possible = []
                for target_name in frame.columns:
                    if target_name == column:
                        continue
                    candidates = ml_risk_findings(frame, target_name)
                    possible.extend(f for f in candidates if f.column == column
                                    and f.evidence.method == evidence.method)
                valid = any(f.evidence.sample_size == evidence.sample_size
                            and f.evidence.metrics == metrics and f.category == finding.category
                            for f in possible)
                verdict = "verified" if valid else "rejected"
                reason = "ML-risk evidence recomputed" if valid else "ML-risk evidence inconsistent"
        elif evidence.method in ("categorical_frequency_ratio", "pandas_sample_skew"):
            if column not in frame.columns or not frame.columns.is_unique:
                verdict, reason = "rejected", "Column missing or ambiguous"
            else:
                # Recompute from raw data with the same descriptive method.
                # Target status changes category labels, not the numerical evidence.
                reference = distribution_findings(frame, column if finding.category == "class_imbalance" else None)
                match = next((f for f in reference if f.column == column
                    and f.evidence.method == evidence.method), None)
                valid = match is not None and evidence.sample_size == match.evidence.sample_size
                if valid:
                    expected = match.evidence.metrics
                    valid = metrics.keys() == expected.keys()
                    if valid:
                        for key, value in expected.items():
                            received = metrics[key]
                            if isinstance(value, (int, float)):
                                if not isinstance(received, (int, float)) or not math.isclose(
                                        float(received), float(value), rel_tol=1e-10, abs_tol=1e-8):
                                    valid = False
                                    break
                            elif received != value:
                                valid = False
                                break
                verdict = "verified" if valid else "rejected"
                reason = "Distribution evidence recomputed" if valid else "Distribution evidence inconsistent"
        elif evidence.method == "tukey_iqr_1_5":
            if column not in frame.columns or not frame.columns.is_unique:
                verdict, reason = "rejected", "Column missing or ambiguous"
            else:
                actual = outlier_summary(frame[column])
                expected = None if actual is None else {k: v for k, v in actual.items() if k != "sample_size"}
                valid = actual is not None and actual["count"] > 0 and evidence.sample_size == actual["sample_size"]
                if valid:
                    for key, value in expected.items():
                        received = metrics.get(key)
                        if not isinstance(received, (int, float)) or not math.isfinite(float(received)) or not math.isclose(float(received), float(value), rel_tol=1e-10, abs_tol=1e-8):
                            valid = False
                            break
                verdict = "verified" if valid else "rejected"
                reason = "IQR fences and counts recomputed" if valid else "Outlier evidence inconsistent"
        elif evidence.method == "pandas_isna_count":
            if column not in frame.columns:
                verdict, reason = "rejected", "Column absent"
            else:
                actual = int(frame[column].isna().sum())
                expected_percent = 100 * actual / len(frame) if len(frame) else 0
                valid = (
                    evidence.sample_size == len(frame)
                    and type(metrics.get("missing_count")) is int
                    and metrics["missing_count"] == actual
                    and isinstance(metrics.get("missing_percentage"), (float, int))
                    and math.isclose(metrics["missing_percentage"], expected_percent, abs_tol=1e-8)
                )
                verdict = "verified" if valid else "rejected"
                reason = "Missing counts recomputed" if valid else "Missing counts or percentage inconsistent"
        elif evidence.method == "chi_square_independence_bh":
            other = metrics.get("other_column")
            if column not in frame.columns or other not in frame.columns or column == other:
                verdict, reason = "rejected", "Invalid referenced columns"
            else:
                observed = frame[other].notna()
                tab = pd.crosstab(frame.loc[observed, other], frame.loc[observed, column].isna())
                if tab.shape[0] < 2 or tab.shape[1] != 2:
                    verdict, reason = "rejected", "Invalid contingency table"
                else:
                    try:
                        _, p, _, expected = chi2_contingency(tab, correction=False)
                        valid = (
                            (expected >= 5).all()
                            and evidence.sample_size == int(observed.sum())
                            and metrics.get("contingency_counts") == tab.astype(int).values.tolist()
                            and isinstance(metrics.get("p_value"), (float, int))
                            and math.isclose(float(metrics["p_value"]), float(p), abs_tol=1e-10)
                            and isinstance(metrics.get("adjusted_p_value"), (float, int))
                            and float(p) <= float(metrics["adjusted_p_value"]) <= 1.0
                        )
                        verdict = "verified" if valid else "rejected"
                        reason = "Association calculation recomputed" if valid else "Association evidence inconsistent"
                    except ValueError:
                        verdict, reason = "rejected", "Statistical assumptions unmet"
        outcomes.append({"finding_id": finding.finding_id, "status": verdict, "reason": reason})
        seen.add(finding.finding_id)
    return outcomes
