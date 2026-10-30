import pandas as pd
import pytest

from datasage.profiling import profile_dataset, profile_findings


def test_duplicate_rows_and_constant_column():
    df = pd.DataFrame({"constant": [7, 7, 7], "variable": [1, 1, 2]})
    profile = profile_dataset(df)
    assert profile["duplicate_row_count"] == 1
    assert profile["column_profiles"][0]["is_constant_non_null"] is True
    findings = profile_findings(profile)
    assert {f.category for f in findings} == {"duplicate_rows", "constant_feature"}
    assert all(f.evidence.method for f in findings)


def test_missingness_and_all_null_not_constant():
    df = pd.DataFrame({"empty": [None, None], "a": [1, None]})
    profile = profile_dataset(df)
    assert profile["column_profiles"][0]["null_percentage"] == 100
    assert profile["column_profiles"][0]["is_constant_non_null"] is False
    assert profile["column_profiles"][1]["null_count"] == 1


def test_empty_rows_have_undefined_percentages():
    profile = profile_dataset(pd.DataFrame({"x": pd.Series(dtype=float)}))
    assert profile["duplicate_row_percentage"] is None
    assert profile["column_profiles"][0]["null_percentage"] is None
    assert profile_findings(profile) == []


def test_duplicate_column_names_rejected():
    frame = pd.DataFrame([[1, 2]], columns=["x", "x"])
    with pytest.raises(ValueError, match="Duplicate column"):
        profile_dataset(frame)


def test_findings_are_candidates_not_data_mutations():
    frame = pd.DataFrame({"x": [1, 1]})
    original = frame.copy(deep=True)
    profile_findings(profile_dataset(frame))
    pd.testing.assert_frame_equal(frame, original)
