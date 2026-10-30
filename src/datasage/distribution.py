"""Conservative within-dataset distribution diagnostics.

Descriptive imbalance is not proof of unfairness or poor ML performance.
No train/test drift is inferred without an explicit split.
"""
from __future__ import annotations

import math
import pandas as pd
from datasage.models import Evidence, Finding


def distribution_findings(frame: pd.DataFrame, target_column: str | None = None) -> list[Finding]:
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names cannot be analyzed")
    if target_column is not None and target_column not in frame.columns:
        raise ValueError("Unknown target column")
    findings: list[Finding] = []
    for idx, name in enumerate(frame.columns, 1):
        series = frame[name].dropna()
        n = len(series)
        if n < 20:
            continue
        unique = int(series.nunique())
        if 2 <= unique <= 20 and (not pd.api.types.is_numeric_dtype(series) or name == target_column):
            counts = series.astype(str).value_counts()
            if len(counts) != unique:
                # String conversion collisions would make evidence ambiguous.
                continue
            top = int(counts.max())
            bottom = int(counts.min())
            ratio = top / bottom
            if ratio < 3:
                continue
            is_target = name == target_column
            findings.append(Finding(
                finding_id=f"DIST-CAT-{idx:03d}", agent="distribution_agent",
                category="class_imbalance" if is_target else "categorical_representation",
                column=str(name),
                title="Uneven category frequencies",
                description="Observed category counts differ substantially.",
                evidence=Evidence(method="categorical_frequency_ratio", sample_size=n,
                    metrics={"counts": counts.to_dict(), "max_min_ratio": ratio, "non_null_count": n}),
                severity="medium" if is_target else "low", confidence="high",
                limitations=["Unequal category frequencies can be expected or intentional.",
                             "The ratio does not demonstrate bias, fairness violations or model failure.",
                             "Null values are excluded and no train/test split is assumed."],
                recommendations=["Check sampling design and the intended prediction task before rebalancing."],
            ))
        elif pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numbers = pd.to_numeric(series, errors="coerce")
            numbers = numbers[numbers.map(math.isfinite)]
            if len(numbers) < 20 or numbers.nunique() < 4:
                continue
            skewness = float(numbers.skew())
            if math.isfinite(skewness) and abs(skewness) >= 1.0:
                findings.append(Finding(
                    finding_id=f"DIST-SKEW-{idx:03d}", agent="distribution_agent",
                    category="skewed_distribution", column=str(name),
                    title="Asymmetric numeric distribution",
                    description="The sample shows substantial numerical skewness.",
                    evidence=Evidence(method="pandas_sample_skew", sample_size=len(numbers),
                                      metrics={"skewness": skewness}),
                    severity="low", confidence="medium",
                    limitations=["Sample skewness does not establish a dataset defect.",
                                 "Nonfinite and null values are excluded."],
                    recommendations=["Inspect the distribution before considering transformations."],
                ))
    return findings
