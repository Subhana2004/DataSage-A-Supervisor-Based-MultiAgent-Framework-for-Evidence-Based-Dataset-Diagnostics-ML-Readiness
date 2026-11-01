from types import SimpleNamespace
import pandas as pd

from datasage.history_store import save_investigation, list_investigations
from datasage.llm_explanations import explain_with_llm


def test_explicit_saved_history(tmp_path):
    report = {"report_version": "1.0", "summary": {"verified": 1}}
    path = save_investigation(report, "../../upload.csv", "Churn", tmp_path)
    assert path.parent == tmp_path
    assert ".." not in path.name
    records = list_investigations(tmp_path)
    assert len(records) == 1
    assert records[0]["report"] == report
    assert records[0]["source_filename"] == "../../upload.csv"


def test_missing_history_empty(tmp_path):
    assert list_investigations(tmp_path / "missing") == []


def test_opt_in_llm_uses_only_finding_evidence():
    captured = {}
    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Observation: unequal frequencies."))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = explain_with_llm({"category": "class_imbalance", "column": "PhoneService",
        "evidence": {"metrics": {"counts": {"Yes": 90, "No": 10}}},
        "validation_status": "verified"}, client=client, model="fake-model")
    assert result.startswith("Observation:")
    assert "PhoneService" in captured["messages"][1]["content"]
    assert captured["model"] == "fake-model"
