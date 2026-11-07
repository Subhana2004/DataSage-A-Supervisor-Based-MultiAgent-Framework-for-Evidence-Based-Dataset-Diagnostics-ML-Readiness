"""Studio UI regression checks: synthetic onboarding and unchanged diagnostics."""
from io import BytesIO

import pandas as pd

from datasage.dashboard import analyze_upload, make_sample_csv
from datasage.ui_theme import STYLE


def test_synthetic_sample_is_valid_small_csv():
    raw = make_sample_csv()
    frame = pd.read_csv(BytesIO(raw))
    assert 0 < len(raw) < 2 * 1024 * 1024
    assert len(frame) == 80
    assert "churn" in frame.columns
    assert frame.isna().any().any()
    assert int(frame.duplicated().sum()) >= 1


def test_sample_runs_full_existing_workflow():
    report = analyze_upload(make_sample_csv(), "datasage_synthetic_customers.csv", "churn")
    assert report["report_version"] == "1.0"
    assert report["dataset_metadata"]["rows"] == 80
    assert any(
        finding["category"] == "target_proxy_candidate"
        for group in report["findings"].values()
        for finding in group
    )
    assert "follow_up_plan" in report
    assert "follow_up_investigations" in report


def test_theme_is_static_and_contains_responsive_rules():
    assert ".ds-hero" in STYLE
    assert "prefers-reduced-motion" in STYLE
    assert "max-width:780px" in STYLE
