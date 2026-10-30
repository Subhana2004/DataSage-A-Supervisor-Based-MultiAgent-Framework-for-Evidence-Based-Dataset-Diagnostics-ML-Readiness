"""Conservative IQR-based outlier candidates on sufficiently populated numeric columns."""
from __future__ import annotations

import numpy as np
import pandas as pd

from datasage.models import Evidence, Finding


def outlier_summary(series: pd.Series) -> dict | None:
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return None
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) < 20 or vals.nunique() < 4:
        return None
    q1, q3 = float(vals.quantile(.25)), float(vals.quantile(.75))
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        return None
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    count = int(((vals < lower) | (vals > upper)).sum())
    return {"sample_size": len(vals), "q1": q1, "q3": q3, "lower": lower,
            "upper": upper, "count": count, "percentage": 100 * count / len(vals)}


def anomaly_findings(frame: pd.DataFrame) -> list[Finding]:
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names are ambiguous")
    findings = []
    for idx, col in enumerate(frame.columns, 1):
        summary = outlier_summary(frame[col])
        if summary is None or not summary["count"]:
            continue
        findings.append(Finding(
            finding_id=f"ANOM-IQR-{idx:03d}", agent="anomaly_agent",
            category="statistical_outlier", column=str(col),
            title="Potential IQR outliers",
            description="Some observed numeric values lie outside Tukey's 1.5×IQR fences.",
            evidence=Evidence(method="tukey_iqr_1_5", sample_size=summary["sample_size"],
                              metrics={k: v for k, v in summary.items() if k != "sample_size"}),
            severity="low", confidence="medium",
            limitations=["Outliers may be legitimate; IQR does not establish invalidity.",
                         "Non-finite and missing observations are excluded.",
                         "Skewed and multimodal distributions may yield misleading fences."],
            recommendations=["Inspect domain ranges and distribution before changing or excluding values."],
        ))
    return findings
