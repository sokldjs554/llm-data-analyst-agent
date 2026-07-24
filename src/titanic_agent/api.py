"""시스템 연계용 REST API (FastAPI).

설계 원칙:
- LLM이 필요 없는 기능(/predict, /dataset/summary)은 LLM 없이도 동작한다.
  → 기존/신규 시스템이 ML 예측만 연계하는 경우 API 키 없이 사용 가능.
- /chat은 에이전트를 통해 자연어 질의를 처리하며, session_id로 멀티턴을 지원한다.
- 상세 명세는 docs/03-api-spec.md 참고.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .agent import TitanicAgent, ToolCallRecord
from .config import Settings
from .data import dataset_overview, load_dataset
from .llm import AnthropicLLM, LLMClient
from .ml import SurvivalModel
from .tools import ToolExecutor

MAX_SESSIONS = 100  # 인메모리 세션 상한 (초과 시 오래된 세션부터 제거)


# ── 요청/응답 모델 ────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="자연어 질문")
    session_id: str | None = Field(
        default=None, description="멀티턴 대화를 이어갈 세션 ID (생략 시 새 세션 발급)"
    )


class ToolCallInfo(BaseModel):
    name: str
    input: dict[str, Any]
    is_error: bool


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    tool_calls: list[ToolCallInfo]
    stop_reason: str


class PredictRequest(BaseModel):
    pclass: Literal[1, 2, 3] = Field(..., description="객실 등급")
    sex: Literal["male", "female"]
    age: float = Field(..., ge=0, le=120)
    sibsp: int = Field(default=0, ge=0)
    parch: int = Field(default=0, ge=0)
    fare: float | None = Field(default=None, ge=0, description="생략 시 등급별 중앙값 사용")
    embarked: Literal["S", "C", "Q"] = "S"


class PredictResponse(BaseModel):
    survived: bool
    survival_probability: float
    model_cv_accuracy: float | None


class HealthResponse(BaseModel):
    status: str
    version: str
    model_ready: bool
    llm_available: bool


# ── 앱 팩토리 ─────────────────────────────────────────────────────────
def create_app(settings: Settings | None = None, llm: LLMClient | None = None) -> FastAPI:
    """FastAPI 앱을 생성한다.

    llm 인자는 테스트에서 MockLLM을 주입하기 위한 것으로,
    지정하지 않으면 설정에 따라 AnthropicLLM을 생성한다.
    """
    settings = settings or Settings()
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        df = load_dataset(settings.data_path)
        model = SurvivalModel()
        model.train(df)
        executor = ToolExecutor(df, model)

        agent_llm = llm
        if agent_llm is None and settings.llm_available:
            agent_llm = AnthropicLLM(
                api_key=settings.anthropic_api_key,
                model=settings.model,
                max_tokens=settings.max_tokens,
            )
        state.update(
            df=df,
            model=model,
            executor=executor,
            agent=(
                TitanicAgent(agent_llm, executor, max_turns=settings.max_agent_turns)
                if agent_llm is not None
                else None
            ),
            sessions={},  # session_id → 대화 이력
        )
        yield
        state.clear()

    app = FastAPI(
        title="Titanic Insight Agent API",
        description="LLM 기반 타이타닉 데이터 분석 AI Agent 연계 API",
        version=__version__,
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            model_ready="model" in state,
            llm_available=state.get("agent") is not None,
        )

    @app.get("/api/v1/dataset/summary", tags=["data"])
    def dataset_summary() -> dict:
        return dataset_overview(state["df"])

    @app.post("/api/v1/predict", response_model=PredictResponse, tags=["ml"])
    def predict(req: PredictRequest) -> PredictResponse:
        model: SurvivalModel = state["model"]
        fare = req.fare
        if fare is None:
            df = state["df"]
            fare = float(df.loc[df["Pclass"] == req.pclass, "Fare"].median())
        result = model.predict(
            pclass=req.pclass,
            sex=req.sex,
            age=req.age,
            sibsp=req.sibsp,
            parch=req.parch,
            fare=fare,
            embarked=req.embarked,
        )
        return PredictResponse(
            survived=result.survived,
            survival_probability=result.probability,
            model_cv_accuracy=model.cv_accuracy,
        )

    @app.post("/api/v1/chat", response_model=ChatResponse, tags=["agent"])
    def chat(req: ChatRequest) -> ChatResponse:
        agent: TitanicAgent | None = state.get("agent")
        if agent is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "LLM이 설정되지 않아 자연어 질의를 처리할 수 없습니다. "
                    "ANTHROPIC_API_KEY 환경 변수를 설정하세요. "
                    "(/api/v1/predict 등 ML 엔드포인트는 키 없이 사용 가능합니다)"
                ),
            )
        sessions: dict[str, list] = state["sessions"]
        session_id = req.session_id or uuid.uuid4().hex
        history = sessions.get(session_id)
        if req.session_id is not None and history is None:
            raise HTTPException(status_code=404, detail=f"세션을 찾을 수 없습니다: {session_id}")

        result = agent.run(req.message, history=history)

        sessions[session_id] = result.history
        if len(sessions) > MAX_SESSIONS:  # 단순 FIFO 정리
            sessions.pop(next(iter(sessions)))

        return ChatResponse(
            answer=result.answer,
            session_id=session_id,
            tool_calls=[
                ToolCallInfo(name=t.name, input=t.input, is_error=t.is_error)
                for t in result.tool_calls
            ],
            stop_reason=result.stop_reason,
        )

    return app


# uvicorn titanic_agent.api:app 으로 실행할 때 사용하는 기본 앱
app = create_app()
