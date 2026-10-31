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
from datasage.missingness import missingness_findings


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
        elif evidence.method == "pandas_dataframe_duplicated":
            count = int(frame.duplicated().sum()) if len(frame.columns) else 0
            percent = round(100 * count / len(frame), 2) if len(frame) else None
            valid = (finding.category == "duplicate_rows" and count > 0
                     and evidence.sample_size == len(frame)
                     and type(metrics.get("duplicate_rows")) is int
                     and metrics["duplicate_rows"] == count
                     and type(metrics.get("percentage")) in (float, int)
                     and metrics["percentage"] == percent)
            verdict = "verified" if valid else "rejected"
            reason = "Duplicate-row counts independently recomputed" if valid else "Duplicate-row evidence inconsistent"
        elif evidence.method == "pandas_series_nunique":
            if column not in frame.columns or not frame.columns.is_unique:
                verdict, reason = "rejected", "Column absent or ambiguous"
            else:
                series = frame[column]
                unique = int(series.nunique(dropna=True))
                non_null = int(series.notna().sum())
                nulls = int(series.isna().sum())
                valid = (finding.category == "constant_feature" and non_null > 0 and unique == 1
                         and evidence.sample_size == non_null
                         and type(metrics.get("unique_non_null")) is int
                         and metrics["unique_non_null"] == unique
                         and type(metrics.get("null_count")) is int
                         and metrics["null_count"] == nulls)
                verdict = "verified" if valid else "rejected"
                reason = "Non-null uniqueness independently recomputed" if valid else "Constant-column evidence inconsistent"
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
            # Rebuild the entire eligible hypothesis family, not merely one p-value.
            # This independently checks its multiplicity and BH-adjusted p-value.
            reference = [
                item for item in missingness_findings(frame)
                if item.evidence.method == "chi_square_independence_bh"
                and item.column == column
                and item.evidence.metrics.get("other_column") == metrics.get("other_column")
            ]
            valid = (len(reference) == 1 and finding.category == "missingness_association"
                     and evidence.sample_size == reference[0].evidence.sample_size
                     and metrics.keys() == reference[0].evidence.metrics.keys()) if reference else False
            if valid:
                for key, expected in reference[0].evidence.metrics.items():
                    received = metrics[key]
                    if isinstance(expected, float):
                        if type(received) not in (int, float) or not math.isfinite(float(received)) or not math.isclose(
                                float(received), expected, rel_tol=1e-10, abs_tol=1e-12):
                            valid = False
                            break
                    elif received != expected:
                        valid = False
                        break
            verdict = "verified" if valid else "rejected"
            reason = "Full BH hypothesis-family adjustment recomputed" if valid else "BH association evidence inconsistent"
        outcomes.append({"finding_id": finding.finding_id, "status": verdict, "reason": reason})
        seen.add(finding.finding_id)
    return outcomes
