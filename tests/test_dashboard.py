import json
import pandas as pd
import pytest

from datasage.dashboard import analyze_upload, filter_findings
from datasage.ingestion import MAX_BYTES
from datasage.reporting import render_html


def csv_payload():
    return pd.DataFrame({"category": ["a"] * 40 + ["b"] * 10,
                         "target": [0] * 40 + [1] * 10}).to_csv(index=False).encode()


def test_upload_integration():
    report = analyze_upload(csv_payload(), "example.csv", "target")
    assert report["report_version"] == "1.0"
    assert report["dataset_metadata"]["rows"] == 50
    assert sum(report["summary"].values()) >= 1
    assert json.loads(json.dumps(report))["report_version"] == "1.0"
    assert "<html" in render_html(report)


def test_no_target_and_filtering():
    report = analyze_upload(csv_payload(), "example.csv")
    assert all(x["category"] != "target_proxy_candidate"
               for group in report["findings"].values() for x in group)
    assert all(f["validation_status"] == "verified"
               for f in filter_findings(report, ["verified"]))


def test_reject_extension_and_missing_target():
    with pytest.raises(ValueError, match="Unsupported"):
        analyze_upload(b"hello", "bad.txt")
    with pytest.raises(ValueError, match="Target column absent"):
        analyze_upload(csv_payload(), "ok.csv", "absent")


def test_size_limit():
    with pytest.raises(ValueError, match="50 MiB"):
        analyze_upload(b"x" * (MAX_BYTES + 1), "large.csv")
