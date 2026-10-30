"""Deterministic, read-only dataset profiling.

Column-level observations are recorded as evidence, not claims of unusability.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from datasage.models import Evidence, Finding


def _native(value: Any) -> Any:
    """Convert common numpy scalars into JSON-friendly Python primitives."""
    if hasattr(value, "item"):
        return value.item()
    return value


def profile_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    """Profile schema, missing values, duplicates and constant columns.

    Empty datasets are supported; callers must interpret statistics with care.
    Column names must be unique so findings can unambiguously reference evidence.
    """
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names cannot be profiled unambiguously")

    columns: list[dict[str, Any]] = []
    for col in frame.columns:
        series = frame[col]
        non_null = int(series.notna().sum())
        columns.append({
            "name": str(col),
            "dtype": str(series.dtype),
            "non_null_count": non_null,
            "null_count": int(series.isna().sum()),
            "null_percentage": round(100 * (len(series) - non_null) / len(series), 2)
            if len(series) else None,
            "unique_non_null": int(series.nunique(dropna=True)),
            "is_constant_non_null": bool(non_null > 0 and series.nunique(dropna=True) == 1),
        })

    duplicates = int(frame.duplicated().sum()) if len(frame.columns) else 0
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "duplicate_row_count": duplicates,
        "duplicate_row_percentage": round(100 * duplicates / len(frame), 2)
        if len(frame) else None,
        "column_profiles": columns,
    }


def profile_findings(profile: dict[str, Any]) -> list[Finding]:
    """Generate traceable *candidate* findings from profiling evidence."""
    findings: list[Finding] = []
    rows = int(profile["rows"])
    if profile["duplicate_row_count"]:
        findings.append(Finding(
            finding_id="PROF-DUP-001",
            agent="profiling_agent",
            category="duplicate_rows",
            title="Duplicate rows observed",
            description="Identical rows were observed; they may be legitimate repeated observations.",
            evidence=Evidence(
                method="pandas_dataframe_duplicated",
                sample_size=rows,
                metrics={
                    "duplicate_rows": profile["duplicate_row_count"],
                    "percentage": profile["duplicate_row_percentage"],
                },
            ),
            severity="low",
            confidence="high",
            limitations=["Exact duplicate rows are not necessarily erroneous records."],
            recommendations=["Check whether repeated observations are expected before deduplication."],
        ))
    for index, column in enumerate(profile["column_profiles"], start=1):
        if column["is_constant_non_null"]:
            findings.append(Finding(
                finding_id=f"PROF-CONST-{index:03d}",
                agent="profiling_agent",
                category="constant_feature",
                column=column["name"],
                title="Single observed non-null value",
                description="All observed non-null values in this column are identical.",
                evidence=Evidence(
                    method="pandas_series_nunique",
                    sample_size=column["non_null_count"],
                    metrics={
                        "unique_non_null": column["unique_non_null"],
                        "null_count": column["null_count"],
                    },
                ),
                severity="low",
                confidence="high",
                limitations=[
                    "The column may be meaningful metadata or a label.",
                    "Missing values are excluded from the uniqueness count.",
                ],
                recommendations=["Review the feature's role before excluding it from a model."],
            ))
    return findings
