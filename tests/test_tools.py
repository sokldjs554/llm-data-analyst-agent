"""에이전트 도구 실행기 테스트."""

import json

import pytest

from titanic_agent.tools import ToolError


def run(executor, name, tool_input):
    return json.loads(executor.execute(name, tool_input))


def test_overview(executor):
    result = run(executor, "get_dataset_overview", {})
    assert result["num_rows"] == 891
    assert "Age" in result["missing_values"]


def test_analyze_survival_by_sex(executor):
    result = run(executor, "analyze_survival", {"group_by": "Sex"})
    rates = {g["group"]: g["survival_rate"] for g in result["groups"]}
    # 알려진 값: 여성 약 74.2%, 남성 약 18.9%
    assert abs(rates["female"] - 0.742) < 0.001
    assert abs(rates["male"] - 0.1889) < 0.001


def test_analyze_survival_with_filters(executor):
    # 30세 미만 승객만 대상으로 등급별 생존율
    result = run(
        executor,
        "analyze_survival",
        {"group_by": "Pclass", "filters": [{"column": "Age", "op": "<", "value": 30}]},
    )
    assert result["total_passengers"] < 891
    assert len(result["groups"]) == 3


def test_analyze_survival_rejects_bad_column(executor):
    with pytest.raises(ToolError):
        executor.execute("analyze_survival", {"group_by": "Cabin"})


def test_filters_reject_disallowed_op(executor):
    with pytest.raises(ToolError):
        executor.execute(
            "analyze_survival",
            {"group_by": "Sex", "filters": [{"column": "Age", "op": "in", "value": [1, 2]}]},
        )


def test_filters_coerce_numeric_string(executor):
    # LLM이 수치를 문자열로 보낸 경우에도 방어적으로 변환한다
    result = run(
        executor,
        "analyze_survival",
        {"group_by": "Sex", "filters": [{"column": "Pclass", "op": "==", "value": "1"}]},
    )
    assert result["total_passengers"] == 216  # 1등석 승객 수


def test_statistics_numeric(executor):
    result = run(executor, "get_statistics", {"column": "Age"})
    assert result["type"] == "numeric"
    assert abs(result["mean"] - 29.7) < 0.1


def test_statistics_categorical(executor):
    result = run(executor, "get_statistics", {"column": "Embarked"})
    assert result["type"] == "categorical"
    assert result["value_counts"]["S"] == 644


def test_statistics_empty_filter_result(executor):
    result = run(
        executor,
        "get_statistics",
        {"column": "Age", "filters": [{"column": "Age", "op": ">", "value": 200}]},
    )
    assert "note" in result


def test_predict_tool(executor):
    result = run(
        executor,
        "predict_survival",
        {"pclass": 1, "sex": "female", "age": 30},
    )
    assert 0.0 <= result["survival_probability"] <= 1.0
    # fare 생략 시 1등석 중앙값으로 대치되었는지 확인
    assert result["input"]["fare"] > 0


def test_unknown_tool(executor):
    with pytest.raises(ToolError):
        executor.execute("drop_table", {})


# ── 리뷰에서 발견된 결함의 회귀 테스트 ──────────────────────────────


def test_filters_reject_non_dict_items(executor):
    """LLM이 스키마를 어겨 filters에 문자열을 보내도 크래시 대신 ToolError."""
    with pytest.raises(ToolError):
        executor.execute("analyze_survival", {"group_by": "Sex", "filters": ["Age > 30"]})


def test_neq_filter_excludes_missing_values(executor, df):
    """!= 필터가 결측(NaN) 행을 '값이 다름'으로 포함하지 않아야 한다."""
    result = run(
        executor,
        "get_statistics",
        {"column": "Age", "filters": [{"column": "Age", "op": "!=", "value": 30}]},
    )
    expected = int((df["Age"].notna() & (df["Age"] != 30)).sum())
    assert result["count"] == expected  # 결측 177명이 섞이면 이 값보다 커진다


def test_single_row_statistics_returns_strict_json(executor):
    """단일 행 표본은 std가 NaN → 표준 JSON(null)으로 치환되어야 한다."""
    raw = executor.execute(
        "get_statistics",
        {"column": "Age", "filters": [{"column": "PassengerId", "op": "==", "value": 1}]},
    )
    assert "NaN" not in raw  # RFC 8259 위반 리터럴 금지
    result = json.loads(raw)
    assert result["count"] == 1
    assert result["std"] is None


def test_predict_rejects_out_of_range_inputs(executor):
    """도구 경로도 REST 경로처럼 입력을 검증해야 한다 (조용한 오답 방지)."""
    with pytest.raises(ToolError):
        executor.execute("predict_survival", {"pclass": 5, "sex": "female", "age": 30})
    with pytest.raises(ToolError):
        executor.execute("predict_survival", {"pclass": 1, "sex": "robot", "age": 30})
    with pytest.raises(ToolError):
        executor.execute("predict_survival", {"pclass": 1, "sex": "male", "age": -3})


def test_analyze_survival_reports_missing_group_keys(executor):
    """그룹 키 결측 인원을 명시해 총원-그룹합 모순을 없앤다."""
    result = run(executor, "analyze_survival", {"group_by": "AgeGroup"})
    assert result["excluded_missing_group_key"] == 177
    group_sum = sum(g["passengers"] for g in result["groups"])
    assert group_sum + result["excluded_missing_group_key"] == result["total_passengers"]
