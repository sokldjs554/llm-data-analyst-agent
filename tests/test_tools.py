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
