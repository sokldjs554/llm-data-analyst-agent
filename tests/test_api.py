"""REST API 통합 테스트 (FastAPI TestClient)."""

import pytest
from fastapi.testclient import TestClient

from titanic_agent.api import create_app
from titanic_agent.config import Settings
from titanic_agent.llm import MockLLM, make_final_response, make_tool_use_response


def make_settings() -> Settings:
    return Settings(anthropic_api_key=None)


@pytest.fixture()
def client_no_llm():
    app = create_app(settings=make_settings())
    with TestClient(app) as client:
        yield client


@pytest.fixture()
def client_with_mock_llm():
    llm = MockLLM(
        [
            make_tool_use_response([("t1", "analyze_survival", {"group_by": "Sex"})]),
            make_final_response("여성 생존율은 74.2%입니다."),
        ]
    )
    app = create_app(settings=make_settings(), llm=llm)
    with TestClient(app) as client:
        yield client


def test_health(client_no_llm):
    res = client_no_llm.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["model_ready"] is True
    assert body["llm_available"] is False


def test_dataset_summary(client_no_llm):
    res = client_no_llm.get("/api/v1/dataset/summary")
    assert res.status_code == 200
    assert res.json()["num_rows"] == 891


def test_predict(client_no_llm):
    res = client_no_llm.post(
        "/api/v1/predict",
        json={"pclass": 1, "sex": "female", "age": 30},
    )
    assert res.status_code == 200
    body = res.json()
    assert 0.0 <= body["survival_probability"] <= 1.0


def test_predict_validation(client_no_llm):
    # 잘못된 등급/성별은 422로 거부되어야 한다
    res = client_no_llm.post(
        "/api/v1/predict",
        json={"pclass": 5, "sex": "female", "age": 30},
    )
    assert res.status_code == 422


def test_chat_without_llm_returns_503(client_no_llm):
    res = client_no_llm.post("/api/v1/chat", json={"message": "성별 생존율은?"})
    assert res.status_code == 503
    assert "ANTHROPIC_API_KEY" in res.json()["detail"]


def test_chat_with_mock_llm(client_with_mock_llm):
    res = client_with_mock_llm.post("/api/v1/chat", json={"message": "성별 생존율은?"})
    assert res.status_code == 200
    body = res.json()
    assert "74.2" in body["answer"]
    assert body["session_id"]
    assert body["tool_calls"][0]["name"] == "analyze_survival"
    assert body["stop_reason"] == "end_turn"


def test_chat_unknown_session_returns_404(client_with_mock_llm):
    res = client_with_mock_llm.post(
        "/api/v1/chat", json={"message": "이어서", "session_id": "없는세션"}
    )
    assert res.status_code == 404
