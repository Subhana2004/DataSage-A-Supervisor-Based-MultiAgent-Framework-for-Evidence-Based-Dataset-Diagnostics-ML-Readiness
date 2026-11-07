"""Public Streamlit deployment must not accept oversized or unsupported uploads."""
import pandas as pd
import pytest

from datasage import dashboard


def test_public_demo_accepts_small_csv(monkeypatch):
    monkeypatch.setattr(dashboard, "PUBLIC_DEMO", True)
    raw = pd.DataFrame({"x": [1, 2, 3], "label": [1, 0, 1]}).to_csv(index=False).encode()
    report = dashboard.analyze_upload(raw, "clean.csv")
    assert report["dataset_metadata"]["rows"] == 3


def test_public_demo_rejects_excel_even_if_valid(monkeypatch):
    monkeypatch.setattr(dashboard, "PUBLIC_DEMO", True)
    with pytest.raises(ValueError, match="Public demo accepts CSV"):
        dashboard.analyze_upload(b"fake", "file.xlsx")


def test_public_demo_rejects_large_payload(monkeypatch):
    monkeypatch.setattr(dashboard, "PUBLIC_DEMO", True)
    with pytest.raises(ValueError, match="2 MiB"):
        dashboard.analyze_upload(b"a" * (dashboard.PUBLIC_LIMIT_BYTES + 1), "file.csv")


def test_public_demo_rejects_rows_over_limit(monkeypatch):
    monkeypatch.setattr(dashboard, "PUBLIC_DEMO", True)
    raw = pd.DataFrame({"value": [1] * (dashboard.PUBLIC_MAX_ROWS + 1)}).to_csv(index=False).encode()
    with pytest.raises(ValueError, match="20,000 rows"):
        dashboard.analyze_upload(raw, "oversize.csv")


def test_public_demo_rejects_columns_over_limit(monkeypatch):
    monkeypatch.setattr(dashboard, "PUBLIC_DEMO", True)
    raw = pd.DataFrame({f"col{i}": [i] for i in range(dashboard.PUBLIC_MAX_COLUMNS + 1)}).to_csv(index=False).encode()
    with pytest.raises(ValueError, match="100 columns"):
        dashboard.analyze_upload(raw, "wide.csv")
