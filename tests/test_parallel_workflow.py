import pandas as pd
from datasage.workflow import build_workflow


def test_parallel_core_join_and_no_target(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1] * 30, "b": list(range(30))}).to_csv(path, index=False)
    result = build_workflow().invoke({"dataset_path": str(path)})
    for name in ("profiling", "missingness", "anomaly", "distribution", "validation", "reporting"):
        assert result["completed_agents"].count(name) == 1
    assert "ml_risk" not in result["completed_agents"]
    assert result["status"] == "report_complete"
    assert any(x["category"] == "constant_feature"
               for items in result["final_report"]["findings"].values() for x in items)


def test_parallel_core_join_with_target(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"target": [0, 1] * 30, "copy": [0, 1] * 30,
                  "unique": list(range(60))}).to_csv(path, index=False)
    result = build_workflow().invoke({"dataset_path": str(path), "target_column": "target"})
    assert result["completed_agents"].count("ml_risk") == 1
    assert result["completed_agents"].count("validation") == 1
    assert result["investigation_plan"].index("ml_risk") < result["investigation_plan"].index("validation")
    assert result["status"] == "report_complete"
