"""LLM 클라이언트 추상화.

에이전트 루프(agent.py)가 특정 SDK에 직접 의존하지 않도록 얇은 인터페이스를 둔다.
- AnthropicLLM : 실제 Claude API 호출 (프로덕션)
- MockLLM     : 사전 정의된 응답 재생 (오프라인 테스트/데모)

raw_content는 제공자가 반환한 원본 콘텐츠 블록을 그대로 보관해,
멀티턴 대화에서 어시스턴트 턴을 손실 없이 되돌려 보낼 수 있게 한다
(Claude의 thinking 블록 보존 요구사항 대응).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class TextBlock:
    text: str


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    """에이전트 루프가 소비하는 정규화된 응답."""

    stop_reason: str  # "end_turn" | "tool_use" | "max_tokens" | "pause_turn" | "refusal"
    text_blocks: list[TextBlock] = field(default_factory=list)
    tool_use_blocks: list[ToolUseBlock] = field(default_factory=list)
    raw_content: Any = None  # 대화 이력에 그대로 되돌려 보낼 원본 콘텐츠

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.text_blocks).strip()


class LLMClient(Protocol):
    """에이전트가 사용하는 LLM 인터페이스."""

    def create_message(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse: ...


class AnthropicLLM:
    """Claude Messages API 기반 구현."""

    def __init__(self, api_key: str, model: str, max_tokens: int = 16000) -> None:
        import anthropic  # 선택적 의존성: 테스트는 MockLLM만으로 동작

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def create_message(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            tools=tools,
            thinking={"type": "adaptive"},
        )
        text_blocks = [TextBlock(b.text) for b in response.content if b.type == "text"]
        tool_use_blocks = [
            ToolUseBlock(id=b.id, name=b.name, input=dict(b.input))
            for b in response.content
            if b.type == "tool_use"
        ]
        return LLMResponse(
            stop_reason=response.stop_reason,
            text_blocks=text_blocks,
            tool_use_blocks=tool_use_blocks,
            raw_content=response.content,
        )


class MockLLM:
    """사전 정의된 LLMResponse 목록을 순서대로 재생하는 테스트용 구현.

    API 키 없이 에이전트 루프 전체(도구 호출 → 결과 반영 → 최종 응답)를
    결정적으로 검증할 수 있다. docs/04-test-strategy.md 참고.
    """

    def __init__(self, script: list[LLMResponse]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []  # 검증용 호출 기록

    def create_message(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        # messages는 호출 이후에도 에이전트가 이어서 변경하는 리스트이므로
        # 검증이 안정적이도록 스냅샷을 기록한다
        self.calls.append({"system": system, "messages": list(messages), "tools": tools})
        if not self._script:
            raise AssertionError("MockLLM 스크립트가 소진되었습니다. 시나리오를 확인하세요.")
        response = self._script.pop(0)
        if response.raw_content is None:
            # 어시스턴트 턴으로 되돌려 보낼 원본 콘텐츠를 API 형식으로 구성
            raw: list[dict[str, Any]] = [
                {"type": "text", "text": b.text} for b in response.text_blocks
            ]
            raw += [
                {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
                for b in response.tool_use_blocks
            ]
            response.raw_content = raw
        return response


def make_tool_use_response(
    tool_calls: list[tuple[str, str, dict[str, Any]]],
    text: str = "",
) -> LLMResponse:
    """테스트 시나리오 작성을 돕는 헬퍼. tool_calls: [(id, name, input), ...]"""
    return LLMResponse(
        stop_reason="tool_use",
        text_blocks=[TextBlock(text)] if text else [],
        tool_use_blocks=[ToolUseBlock(id=i, name=n, input=inp) for i, n, inp in tool_calls],
    )


def make_final_response(text: str) -> LLMResponse:
    return LLMResponse(stop_reason="end_turn", text_blocks=[TextBlock(text)])
