import pandas as pd
from datasage.external_evaluation import compare_frame, pandas_baseline


def test_baseline_deterministic():
    frame = pd.DataFrame({"constant": [1, 1, 1], "missing": [1, None, 2]})
    assert ("missing_data", "missing") in pandas_baseline(frame)
    assert ("constant_feature", "constant") in pandas_baseline(frame)


def test_comparison_without_ground_truth():
    frame = pd.DataFrame({"constant": [1] * 25, "x": list(range(25))})
    result = compare_frame(frame)
    assert ["constant_feature", "constant"] in result["agreement"]
    assert result["baseline_only"] == []
    assert result["datasage_only"] == []


def test_duplicate_and_missing_agreement():
    frame = pd.DataFrame({"a": [1] * 25, "b": [None] * 25})
    result = compare_frame(frame)
    assert ["duplicate_rows", ""] in result["agreement"]
    assert ["missing_data", "b"] in result["agreement"]
