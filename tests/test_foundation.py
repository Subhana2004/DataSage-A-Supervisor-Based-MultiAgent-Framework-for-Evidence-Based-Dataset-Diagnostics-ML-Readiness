from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from datasage.ingestion import load_dataset
from datasage.models import Evidence, Finding
from datasage.workflow import build_workflow


def test_csv_ingestion_and_fingerprint(tmp_path: Path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"feature": [1, 2], "target": [0, 1]}).to_csv(path, index=False)
    frame, metadata = load_dataset(path)
    assert frame.shape == (2, 2)
    assert metadata.schema_ == {"feature": "int64", "target": "int64"}
    assert len(metadata.fingerprint) == 64


def test_unsupported_type_rejected(tmp_path: Path):
    path = tmp_path / "sample.txt"
    path.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported"):
        load_dataset(path)


def test_finding_schema_requires_evidence():
    with pytest.raises(ValidationError):
        Finding(
            finding_id="MISS-001", agent="missingness", category="missing_data",
            title="Missing data", description="Example", severity="medium",
            confidence="high",
        )
    finding = Finding(
        finding_id="MISS-001", agent="missingness", category="missing_data",
        title="Missing data", description="Example",
        evidence=Evidence(method="count", metrics={"missing": 2}, sample_size=10),
        severity="medium", confidence="high",
    )
    assert finding.evidence.metrics["missing"] == 2


def test_initial_supervisor_plan(tmp_path: Path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"x": [1], "y": [0]}).to_csv(path, index=False)
    result = build_workflow().invoke({"dataset_path": str(path), "target_column": "y"})
    assert result["status"] == "planned"
    assert result["investigation_plan"] == ["profiling", "ml_risk"]
    assert result["dataset_metadata"]["rows"] == 1
