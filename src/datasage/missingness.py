"""Missingness diagnostics: descriptive counts and exploratory associations.

No observational test here can prove MCAR, MAR, or MNAR.
"""
from __future__ import annotations

import pandas as pd
from scipy.stats import chi2_contingency

from datasage.models import Evidence, Finding


def missingness_findings(frame: pd.DataFrame) -> list[Finding]:
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names are ambiguous")
    if not len(frame):
        return []
    findings: list[Finding] = []
    candidates: list[tuple[str, str, float, int, list[list[int]]]] = []
    for index, name in enumerate(frame.columns, start=1):
        mask = frame[name].isna()
        count = int(mask.sum())
        if count == 0:
            continue
        findings.append(Finding(
            finding_id=f"MISS-RATE-{index:03d}", agent="missingness_agent",
            category="missing_data", column=str(name),
            title="Missing observations detected",
            description="Some observations do not contain a value.",
            evidence=Evidence(method="pandas_isna_count", sample_size=len(frame),
                              metrics={"missing_count": count, "missing_percentage": 100 * count / len(frame)}),
            severity="low", confidence="high",
            limitations=["Missingness alone does not establish data corruption or a missingness mechanism."],
            recommendations=["Investigate collection context before imputation or deletion."],
        ))
        if count == len(frame):
            continue
        for other in frame.columns:
            if other == name:
                continue
            series = frame[other]
            # Conservative: avoid treating continuous/identifier features as categories.
            if series.nunique(dropna=True) < 2 or series.nunique(dropna=True) > 10:
                continue
            observed = series.notna()
            if observed.sum() < 20:
                continue
            table = pd.crosstab(series[observed], mask[observed])
            if table.shape[0] < 2 or table.shape[1] < 2:
                continue
            try:
                statistic, p_value, _, expected = chi2_contingency(table, correction=False)
            except ValueError:
                continue
            if (expected < 5).any():
                continue
            candidates.append((str(name), str(other), float(p_value), int(observed.sum()),
                               table.astype(int).values.tolist()))
    # Benjamini-Hochberg controls FDR for the family actually tested.
    ordered = sorted(enumerate(candidates), key=lambda pair: pair[1][2])
    m = len(ordered)
    adjusted = [1.0] * m
    running = 1.0
    for position in range(m - 1, -1, -1):
        original_index, item = ordered[position]
        running = min(running, item[2] * m / (position + 1))
        adjusted[original_index] = running
    for index, (name, other, p_value, n, counts) in enumerate(candidates):
        findings.append(Finding(
            finding_id=f"MISS-ASSOC-{index + 1:03d}", agent="missingness_agent",
            category="missingness_association", column=name,
            title="Exploratory missingness association",
            description=f"Missingness in {name} was tested against observed categories of {other}.",
            evidence=Evidence(method="chi_square_independence_bh", sample_size=n,
                metrics={"other_column": other, "p_value": p_value,
                         "adjusted_p_value": adjusted[index], "family_size": m,
                         "contingency_counts": counts}),
            severity="medium" if adjusted[index] < 0.05 else "low",
            confidence="medium" if adjusted[index] < 0.05 else "low",
            limitations=["Association is not causation or proof of MAR, MNAR or MCAR.",
                         "Rows missing the comparison feature were excluded.",
                         "Exploratory tests can remain confounded."],
            recommendations=["Review the collection mechanism and confirm on held-out data."],
        ))
    return findings
