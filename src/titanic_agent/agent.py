"""데이터 분석 AI 에이전트 루프.

동작 방식 (수동 tool-use 루프):
1. 사용자 질문 + 시스템 프롬프트 + 도구 정의를 LLM에 전달
2. stop_reason == "tool_use" 이면 요청된 도구를 전부 실행하고
   결과를 하나의 user 턴(tool_result 목록)으로 되돌려 보냄
3. LLM이 최종 텍스트 응답(end_turn)을 낼 때까지 반복 (최대 max_turns)

SDK의 tool runner 대신 수동 루프를 채택한 이유(감사 로그, Mock 주입 용이성 등)는
docs/02-agent-strategy.md에 정리했다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .llm import LLMClient
from .tools import TOOL_DEFINITIONS, ToolError, ToolExecutor

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """당신은 타이타닉 승객 데이터셋(891명)을 분석하는 데이터 분석 AI 에이전트입니다.

역할:
- 사용자의 자연어 질문을 해석해 적절한 분석 도구를 호출하고, 결과를 근거로 답변합니다.
- 답변에는 도구가 반환한 실제 수치를 인용합니다. 수치를 추측하거나 지어내지 않습니다.
- 데이터로 확인할 수 없는 내용은 확인할 수 없다고 명확히 말합니다.

도구 사용 지침:
- 데이터 관련 질문에는 반드시 도구를 먼저 호출해 확인한 뒤 답합니다. 기억에 의존해 답하지 않습니다.
- 비교 질문(예: 성별 vs 등급)은 필요한 도구를 각각 호출해 근거를 모두 수집합니다.
- 가정형 질문("~라면 생존했을까?")에는 predict_survival 도구를 사용하고,
  결과가 확률 기반 추정임을 함께 안내합니다.

응답 스타일:
- 한국어로 답합니다. 핵심 결론을 먼저 말하고, 근거 수치를 이어서 제시합니다.
- 생존율은 백분율로 표기합니다(예: 74.2%).
- 데이터셋 범위(1912년 타이타닉 승객 891명 표본)를 벗어난 일반화는 주의 문구를 덧붙입니다."""


@dataclass
class ToolCallRecord:
    """API 응답/감사 로그용 도구 호출 기록."""

    name: str
    input: dict[str, Any]
    output: str
    is_error: bool = False


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "end_turn"


class TitanicAgent:
    """자연어 질문을 받아 도구 기반 분석을 수행하는 에이전트."""

    def __init__(
        self,
        llm: LLMClient,
        executor: ToolExecutor,
        max_turns: int = 10,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.llm = llm
        self.executor = executor
        self.max_turns = max_turns
        self.system_prompt = system_prompt

    def run(
        self,
        user_message: str,
        history: list[dict[str, Any]] | None = None,
    ) -> AgentResult:
        """단일 질문을 처리한다. history를 넘기면 멀티턴 대화를 이어간다."""
        messages: list[dict[str, Any]] = list(history or [])
        messages.append({"role": "user", "content": user_message})
        tool_calls: list[ToolCallRecord] = []

        for turn in range(self.max_turns):
            response = self.llm.create_message(
                system=self.system_prompt,
                messages=messages,
                tools=TOOL_DEFINITIONS,
            )
            messages.append({"role": "assistant", "content": response.raw_content})

            if response.stop_reason == "tool_use":
                results = []
                for block in response.tool_use_blocks:
                    record = self._execute_tool(block.name, block.input)
                    tool_calls.append(record)
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": record.output,
                            **({"is_error": True} if record.is_error else {}),
                        }
                    )
                # 병렬 도구 호출 결과는 반드시 하나의 user 턴으로 모아 보낸다
                messages.append({"role": "user", "content": results})
                continue

            if response.stop_reason == "pause_turn":
                # 서버 측 처리가 일시 중단된 턴: 이력을 유지한 채 재요청하면 이어서 진행된다
                continue

            if response.stop_reason == "refusal":
                return AgentResult(
                    answer="죄송합니다. 해당 요청은 처리할 수 없습니다.",
                    tool_calls=tool_calls,
                    history=messages,
                    stop_reason="refusal",
                )

            # end_turn / max_tokens → 현재 텍스트를 최종 답변으로 반환
            return AgentResult(
                answer=response.text,
                tool_calls=tool_calls,
                history=messages,
                stop_reason=response.stop_reason,
            )

        logger.warning("에이전트가 max_turns(%d)에 도달했습니다.", self.max_turns)
        return AgentResult(
            answer="분석 단계가 예상보다 길어져 중단했습니다. 질문을 좁혀서 다시 시도해 주세요.",
            tool_calls=tool_calls,
            history=messages,
            stop_reason="max_turns_exceeded",
        )

    def _execute_tool(self, name: str, tool_input: dict[str, Any]) -> ToolCallRecord:
        """도구를 실행하고 기록을 남긴다. 오류는 is_error 결과로 변환해 LLM이 정정하게 한다."""
        try:
            output = self.executor.execute(name, tool_input)
            logger.info("tool=%s input=%s → ok", name, tool_input)
            return ToolCallRecord(name=name, input=tool_input, output=output)
        except ToolError as exc:
            logger.warning("tool=%s input=%s → error: %s", name, tool_input, exc)
            return ToolCallRecord(name=name, input=tool_input, output=str(exc), is_error=True)
