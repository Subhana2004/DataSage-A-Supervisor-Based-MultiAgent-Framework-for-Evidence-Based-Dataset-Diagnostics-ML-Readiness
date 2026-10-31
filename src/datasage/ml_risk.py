"""Conservative ML-risk screening; signals are not proof of leakage."""
from __future__ import annotations
import re
import pandas as pd
from datasage.models import Evidence, Finding

_ID_PATTERN = re.compile(r"(^id$|_id$|^id_|identifier|uuid|guid|record.?number)", re.I)


def ml_risk_findings(frame: pd.DataFrame, target_column: str, prediction_context: str | None = None) -> list[Finding]:
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names are ambiguous")
    if target_column not in frame.columns:
        raise ValueError("Unknown target column")
    findings: list[Finding] = []
    target = frame[target_column]
    if len(frame) < 20:
        return []
    for i, col in enumerate(frame.columns, 1):
        if col == target_column:
            continue
        series = frame[col]
        n = int(series.notna().sum())
        if n < 20:
            continue
        uniq = int(series.nunique(dropna=True))
        ratio = uniq / n
        if _ID_PATTERN.search(str(col)) and ratio >= 0.95:
            findings.append(Finding(
                finding_id=f"MLR-ID-{i:03d}", agent="ml_risk_agent",
                category="identifier_candidate", column=str(col),
                title="Identifier-like feature",
                description="A name resembling an identifier has nearly unique observed values.",
                evidence=Evidence(method="identifier_name_uniqueness", sample_size=n,
                                  metrics={"unique_non_null": uniq, "unique_ratio": ratio}),
                severity="low", confidence="medium",
                limitations=["Identifier-like columns can still be legitimate predictors.",
                             "Naming and uniqueness cannot establish target leakage."],
                recommendations=["Check whether the identifier would exist at prediction time and generalize to unseen entities."],
            ))
        # Exact matching observations are evidence of a risky relationship,
        # but may reflect an intentionally replicated variable.
        mask = series.notna() & target.notna()
        paired = int(mask.sum())
        if paired >= 20:
            agreement = float((series[mask].astype("string") == target[mask].astype("string")).mean())
            if agreement >= 0.99 and target[mask].nunique() >= 2:
                findings.append(Finding(
                    finding_id=f"MLR-MATCH-{i:03d}", agent="ml_risk_agent",
                    category="target_proxy_candidate", column=str(col),
                    title="Feature nearly matches target",
                    description="The feature agrees with the target on nearly all jointly observed rows.",
                    evidence=Evidence(method="paired_target_agreement", sample_size=paired,
                                      metrics={"agreement": agreement, "paired_rows": paired}),
                    severity="medium", confidence="high",
                    limitations=["Matching the target does not prove leakage or establish prediction-time availability.",
                                 "String comparison may miss transformed proxies."],
                    recommendations=["Confirm that this feature is available when predictions are made and is not generated from the outcome."],
                ))
    return findings
