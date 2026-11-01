"""Deterministic, transparent human-readable summaries of diagnostic evidence."""
from __future__ import annotations


def explain_finding(item: dict) -> dict[str, str]:
    """Describe measured facts separately from possible consequences."""
    metrics = item.get("evidence", {}).get("metrics", {})
    category = item.get("category")
    column = str(item.get("column") or "the dataset")
    if category == "duplicate_rows":
        count = metrics.get("duplicate_rows", "?")
        percent = metrics.get("percentage", "?")
        fact = f"{count} rows ({percent}%) exactly match another row."
        meaning = "These may be genuine repeated observations, not necessarily mistakes."
        next_step = "Check record identifiers and collection rules before deleting anything."
    elif category == "missing_data":
        count = metrics.get("missing_count", "?")
        percent = metrics.get("missing_percentage", "?")
        fact = f"{column} has {count} missing values ({float(percent):.1f}%)." if isinstance(percent, (int, float)) else f"{column} has {count} missing values."
        meaning = "The reason for missing values is not established by counting them."
        next_step = "Review how this field was collected before choosing an imputation strategy."
    elif category in ("categorical_representation", "class_imbalance"):
        counts = metrics.get("counts", {})
        if isinstance(counts, dict) and counts:
            fact = f"In {column}: " + ", ".join(f"{key}: {value}" for key, value in counts.items()) + "."
        else:
            fact = f"Category frequencies in {column} are uneven."
        meaning = "Unequal frequencies may be normal and are not by themselves proof of bias or poor predictions."
        next_step = "Check whether this distribution matters for your prediction goal."
    elif category == "constant_feature":
        fact = f"{column} has only one distinct observed non-missing value."
        meaning = "It might be uninformative for prediction, but could be meaningful metadata."
        next_step = "Check whether this column has an intended role before removing it."
    elif category == "statistical_outlier":
        fact = f"{column} contains values outside a statistical IQR range."
        meaning = "An unusually large or small value is not automatically an error."
        next_step = "Inspect the unusual values and their real-world meaning."
    elif category == "target_proxy_candidate":
        fact = f"{column} closely matches the selected prediction target."
        meaning = "This suggests a possible shortcut, but does not prove target leakage."
        next_step = "Confirm whether the feature is available at the moment of prediction."
    else:
        fact = str(item.get("description") or item.get("title") or "A pattern was observed.")
        meaning = "Interpret the observation in the context of your data and intended use."
        next_step = "; ".join(item.get("recommendations", [])) or "Review the underlying evidence."
    return {"observation": fact, "meaning": meaning, "next_step": next_step}


def overall_summary(report: dict) -> str:
    values = report.get("summary", {})
    verified = int(values.get("verified", 0))
    rejected = int(values.get("rejected", 0))
    unverified = int(values.get("not_verified", 0))
    return (f"The analysis produced {verified} numerically verified observation(s), "
            f"{rejected} rejected finding(s), and {unverified} finding(s) requiring verification. "
            "Verified means that the calculation was checked, not that a harmful data problem was proven.")
