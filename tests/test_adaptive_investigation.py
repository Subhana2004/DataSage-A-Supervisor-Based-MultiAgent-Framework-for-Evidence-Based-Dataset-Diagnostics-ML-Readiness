"""Regression coverage for evidence-seeking diagnostics and bounded LLM planning."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from datasage.adaptive_investigation import (
    plan_followups, investigate_missingness, investigate_temporal_availability,
    run_followups, structured_context,
)
from datasage.workflow import build_workflow


def context():
    return {
        "prediction_time_column": "prediction_at",
        "feature_time_columns": {"outcome_proxy": "outcome_proxy_recorded_at"},
        "missing_reason_columns": {"income": "income_missing_reason"},
    }


def dataset():
    n = 40
    return pd.DataFrame({
        "target": [i % 2 for i in range(n)],
        "outcome_proxy": [i % 2 for i in range(n)],
        "prediction_at": ["2026-11-01T12:00:00Z"] * n,
        "outcome_proxy_recorded_at": ["2026-11-02T12:00:00Z"] * 20
                                     + ["2026-10-31T12:00:00Z"] * 20,
        "income": [None] * 10 + list(range(30)),
        "income_missing_reason": ["not_collected"] * 8 + [None] * 32,
    })


def test_context_parsing_remains_compatible_with_free_text():
    assert structured_context("Predict at time of purchase") == {}
    assert structured_context("{not valid") == {}
    assert structured_context(json.dumps(context()))["prediction_time_column"] == "prediction_at"


def test_missingness_documents_sources_without_claiming_causality():
    result = investigate_missingness(dataset(), context())
    income = next(x for x in result["observations"] if x["column"] == "income")
    assert result["status"] == "recorded_explanations"
    assert income["missing_rows"] == 10
    assert income["documented_rows"] == 8
    assert income["unexplained_rows"] == 2
    assert "cannot independently" in result["interpretation"]


def test_temporal_comparison_find_post_prediction_values():
    data = dataset()
    original = data.copy(deep=True)
    result = investigate_temporal_availability(data, context())
    assert result["status"] == "post_cutoff_evidence"
    assert result["observations"][0]["post_cutoff_rows"] == 20
    pd.testing.assert_frame_equal(data, original)


def test_without_time_metadata_does_not_invent_leakage():
    result = investigate_temporal_availability(dataset(), {})
    assert result["status"] == "needs_prediction_timestamps"
    assert result["observations"] == []


def test_supervisor_selects_only_eligible_followups():
    data = dataset()
    findings = [{"category": "target_proxy_candidate", "column": "outcome_proxy"}]
    decision = plan_followups(data, findings, "target", json.dumps(context()))
    assert decision["selected"] == [
        "temporal_availability", "missingness_provenance", "target_proxy_review"
    ]
    result = run_followups(data, findings, json.dumps(context()), decision["selected"])
    assert [x["check"] for x in result] == decision["selected"]


class FakeClient:
    def __init__(self, content):
        self.content = content
        self.sent = []

    @property
    def chat(self):
        return SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.sent.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=self.content)
        )])


def test_llm_planner_is_optional_bounded_and_does_not_receive_raw_rows():
    data = dataset()
    client = FakeClient('{"checks":["temporal_availability"],"reason":"Timing metadata supplied"}')
    selected = plan_followups(data, [], "target", json.dumps(context()), mode="llm", client=client)
    assert selected["source"] == "opt_in_llm"
    assert selected["selected"] == ["temporal_availability"]
    payload = client.sent[0]["messages"][1]["content"]
    assert "not_collected" not in payload
    assert "2026-11-02T12:00:00Z" not in payload
    with pytest.raises(ValueError, match="unapproved"):
        plan_followups(data, [], "target", json.dumps(context()), mode="llm",
                       client=FakeClient('{"checks":["delete_dataset"]}'))


def test_no_target_and_no_missing_values_produces_no_followups(tmp_path):
    file = tmp_path / "clean.csv"
    pd.DataFrame({"x": list(range(40)), "y": list(range(40))}).to_csv(file, index=False)
    state = build_workflow().invoke({"dataset_path": str(file)})
    assert state["final_report"]["follow_up_plan"] == []
    assert state["final_report"]["follow_up_investigations"] == []


def test_end_to_end_followups_and_validation_preserved(tmp_path):
    file = tmp_path / "diagnostic.csv"
    dataset().to_csv(file, index=False)
    state = build_workflow().invoke({
        "dataset_path": str(file), "target_column": "target",
        "prediction_context": json.dumps(context()),
    })
    report = state["final_report"]
    assert state["status"] == "report_complete"
    assert report["follow_up_plan"] == [
        "temporal_availability", "missingness_provenance", "target_proxy_review"
    ]
    assert {x["check"] for x in report["follow_up_investigations"]} == set(report["follow_up_plan"])
    assert any(x["status"] == "post_cutoff_evidence" for x in report["follow_up_investigations"])
    assert "verified" in report["findings"]
    assert state["completed_agents"].count("validation") == 1
