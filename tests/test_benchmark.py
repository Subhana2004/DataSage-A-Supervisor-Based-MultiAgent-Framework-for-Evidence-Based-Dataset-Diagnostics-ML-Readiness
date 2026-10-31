import json

from datasage.benchmark import benchmark_cases, observed_keys, score_benchmark


def test_cases_have_explicit_unique_names():
    cases = benchmark_cases()
    assert len(cases) == 8
    assert len({x[0] for x in cases}) == len(cases)
    assert not next(x for x in cases if x[0] == "clean")[3]


def test_score_is_deterministic_and_serializable():
    left = score_benchmark()
    right = score_benchmark()
    assert left == right
    assert len(left["cases"]) == 6
    assert left["totals"]["tp"] >= 0
    assert left["totals"]["fp"] >= 0
    assert left["totals"]["fn"] >= 0
    json.dumps(left, allow_nan=False)


def test_unverified_does_not_count_as_detection():
    report = {"findings": {"verified": [], "not_verified": [
        {"category": "missing_data", "column": "value"}
    ], "rejected": []}}
    assert observed_keys(report, {"missing_data"}) == set()


def test_wrong_category_not_counted():
    report = {"findings": {"verified": [
        {"category": "unrelated", "column": "x"}
    ], "not_verified": [], "rejected": []}}
    assert observed_keys(report, {"missing_data"}) == set()
