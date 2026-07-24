"""CLI 데모: 터미널에서 에이전트에게 직접 질문한다.

사용법:
    export ANTHROPIC_API_KEY=...
    python scripts/demo.py "여성과 남성의 생존율을 비교해줘"
    python scripts/demo.py            # 대화형 모드

API 키 없이 에이전트 루프만 확인하려면:
    python scripts/demo.py --mock    # 사전 정의된 시나리오 재생
"""

from __future__ import annotations

import sys
from pathlib import Path

# src 레이아웃 패키지를 스크립트에서 바로 임포트하기 위한 경로 추가
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from titanic_agent.agent import AgentResult, TitanicAgent  # noqa: E402
from titanic_agent.config import Settings  # noqa: E402
from titanic_agent.data import load_dataset  # noqa: E402
from titanic_agent.llm import (  # noqa: E402
    AnthropicLLM,
    MockLLM,
    make_final_response,
    make_tool_use_response,
)
from titanic_agent.ml import SurvivalModel  # noqa: E402
from titanic_agent.tools import ToolExecutor  # noqa: E402


def build_agent(settings: Settings, mock: bool = False) -> TitanicAgent:
    df = load_dataset(settings.data_path)
    model = SurvivalModel()
    print(f"[준비] 데이터 {len(df)}건 로드, 모델 학습 중...")
    accuracy = model.train(df)
    print(f"[준비] 모델 학습 완료 (5-fold CV 정확도: {accuracy:.1%})")
    executor = ToolExecutor(df, model)

    if mock:
        llm = MockLLM(
            [
                make_tool_use_response([("demo_1", "analyze_survival", {"group_by": "Sex"})]),
                make_final_response(
                    "성별 생존율 분석 결과입니다. (Mock 데모: 실제 LLM 없이 "
                    "에이전트 루프와 도구 실행을 재생한 결과이며, 수치는 실제 데이터입니다.)"
                ),
            ]
        )
        return TitanicAgent(llm, executor)

    if not settings.llm_available:
        print("오류: ANTHROPIC_API_KEY가 설정되지 않았습니다. --mock 으로 오프라인 데모를 실행할 수 있습니다.")
        raise SystemExit(1)
    llm = AnthropicLLM(settings.anthropic_api_key, settings.model, settings.max_tokens)
    return TitanicAgent(llm, executor)


def print_result(result: AgentResult) -> None:
    for call in result.tool_calls:
        status = "오류" if call.is_error else "완료"
        print(f"  [도구] {call.name}({call.input}) → {status}")
    print(f"\n{result.answer}\n")


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--mock"]
    mock = "--mock" in sys.argv[1:]
    settings = Settings()
    agent = build_agent(settings, mock=mock)

    if mock:
        print("\n[Mock 데모] 질문: 성별 생존율을 알려줘")
        print_result(agent.run("성별 생존율을 알려줘"))
        return

    if args:  # 단일 질문 모드
        print_result(agent.run(" ".join(args)))
        return

    # 대화형 모드 (멀티턴)
    print("\n타이타닉 데이터 분석 에이전트입니다. 질문을 입력하세요. (종료: quit)\n")
    history = None
    while True:
        try:
            question = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"quit", "exit", "종료"}:
            break
        result = agent.run(question, history=history)
        history = result.history
        print_result(result)


if __name__ == "__main__":
    main()
