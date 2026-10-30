"""Independent deterministic checks of diagnostic evidence.

Validation confirms supported calculations, not causal interpretation.
"""
from __future__ import annotations

import math

import pandas as pd
from scipy.stats import chi2_contingency

from datasage.anomaly import outlier_summary
from datasage.models import Finding


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
