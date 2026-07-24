"""에이전트 루프 테스트 (MockLLM 기반 — API 키 불필요, 결정적 실행)."""

import json

from titanic_agent.agent import TitanicAgent
from titanic_agent.llm import MockLLM, make_final_response, make_tool_use_response


def test_single_tool_call_flow(executor):
    """도구 호출 → 결과 반영 → 최종 응답의 기본 흐름."""
    llm = MockLLM(
        [
            make_tool_use_response([("t1", "analyze_survival", {"group_by": "Sex"})]),
            make_final_response("여성 생존율은 74.2%, 남성은 18.9%입니다."),
        ]
    )
    agent = TitanicAgent(llm, executor)
    result = agent.run("성별 생존율을 알려줘")

    assert "74.2" in result.answer
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "analyze_survival"
    assert not result.tool_calls[0].is_error
    # 두 번째 LLM 호출에 도구 결과가 tool_result로 전달되었는지 확인
    second_call_messages = llm.calls[1]["messages"]
    tool_result_turn = second_call_messages[-1]
    assert tool_result_turn["role"] == "user"
    assert tool_result_turn["content"][0]["type"] == "tool_result"
    payload = json.loads(tool_result_turn["content"][0]["content"])
    assert payload["group_by"] == "Sex"


def test_parallel_tool_calls_returned_in_single_turn(executor):
    """병렬 도구 호출 결과가 하나의 user 턴으로 모여 전달되어야 한다."""
    llm = MockLLM(
        [
            make_tool_use_response(
                [
                    ("t1", "analyze_survival", {"group_by": "Sex"}),
                    ("t2", "analyze_survival", {"group_by": "Pclass"}),
                ]
            ),
            make_final_response("성별과 등급 모두 생존율 차이가 뚜렷합니다."),
        ]
    )
    agent = TitanicAgent(llm, executor)
    result = agent.run("성별과 등급별 생존율을 비교해줘")

    assert len(result.tool_calls) == 2
    tool_result_turn = llm.calls[1]["messages"][-1]
    assert len(tool_result_turn["content"]) == 2
    assert {c["tool_use_id"] for c in tool_result_turn["content"]} == {"t1", "t2"}


def test_tool_error_is_returned_to_llm(executor):
    """도구 오류는 예외로 죽지 않고 is_error 결과로 LLM에 전달되어야 한다."""
    llm = MockLLM(
        [
            make_tool_use_response([("t1", "analyze_survival", {"group_by": "Cabin"})]),
            make_final_response("Cabin 기준 분석은 지원하지 않습니다."),
        ]
    )
    agent = TitanicAgent(llm, executor)
    result = agent.run("선실별 생존율")

    assert result.tool_calls[0].is_error
    error_content = llm.calls[1]["messages"][-1]["content"][0]
    assert error_content["is_error"] is True


def test_max_turns_guard(executor):
    """LLM이 도구 호출을 멈추지 않아도 max_turns에서 안전하게 종료된다."""
    script = [
        make_tool_use_response([(f"t{i}", "get_dataset_overview", {})]) for i in range(5)
    ]
    llm = MockLLM(script)
    agent = TitanicAgent(llm, executor, max_turns=3)
    result = agent.run("계속 분석해줘")

    assert result.stop_reason == "max_turns_exceeded"
    assert len(llm.calls) == 3


def test_multiturn_history(executor):
    """이전 대화 이력을 넘기면 이어서 대화할 수 있다."""
    llm = MockLLM(
        [
            make_tool_use_response([("t1", "analyze_survival", {"group_by": "Sex"})]),
            make_final_response("여성 74.2%, 남성 18.9%입니다."),
            make_final_response("네, 여성의 생존율이 약 3.9배 높습니다."),
        ]
    )
    agent = TitanicAgent(llm, executor)
    first = agent.run("성별 생존율은?")
    second = agent.run("여성이 더 높은 거지?", history=first.history)

    assert "3.9배" in second.answer
    # 두 번째 실행의 첫 호출에 이전 대화 이력이 포함되어야 한다
    assert len(llm.calls[2]["messages"]) == len(first.history) + 1
