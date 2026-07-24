# 03. 시스템 연계 API 명세

> 기존/신규 시스템이 AI Agent와 ML 예측 기능을 연계하기 위한 REST API 명세입니다.
> 서버 실행 후 `http://localhost:8000/docs` 에서 Swagger UI로도 확인할 수 있습니다.

## 실행

```bash
uvicorn titanic_agent.api:app --app-dir src --host 0.0.0.0 --port 8000
```

| 환경 변수 | 필수 | 설명 |
|---|---|---|
| `ANTHROPIC_API_KEY` | `/chat` 사용 시 | 미설정이면 `/chat`만 503, 나머지는 정상 동작 |
| `TITANIC_AGENT_MODEL` | ✗ | 기본 `claude-opus-4-8` |
| `TITANIC_DATA_PATH` | ✗ | 기본 `data/titanic.csv` |

서버 기동 시 데이터 로드와 모델 학습(수 초)이 완료된 후 요청을 받기 시작합니다.

## 엔드포인트 요약

| 메서드 | 경로 | LLM | 설명 |
|---|---|---|---|
| GET | `/health` | ✗ | 헬스 체크 (모델·LLM 준비 상태 포함) |
| GET | `/api/v1/dataset/summary` | ✗ | 데이터셋 요약 |
| POST | `/api/v1/predict` | ✗ | 생존 예측 (구조화 입력) |
| POST | `/api/v1/chat` | ✓ | 자연어 질의 (AI Agent, 멀티턴) |

---

## GET /health

```json
{
  "status": "ok",
  "version": "0.1.0",
  "model_ready": true,
  "llm_available": true
}
```

- `model_ready`: ML 파이프라인 학습 완료 여부
- `llm_available`: `/chat` 사용 가능 여부 — 연계 시스템은 이 값으로 기능 노출을 결정

## GET /api/v1/dataset/summary

행/열 수, 컬럼 스키마, 결측치 현황, 전체 생존율, 주요 수치 통계를 반환합니다.

```json
{
  "num_rows": 891,
  "num_columns": 14,
  "columns": {"Survived": "int64", "Pclass": "int64", "...": "..."},
  "missing_values": {"Age": 177, "Cabin": 687, "Embarked": 2},
  "survival_rate": 0.3838,
  "numeric_summary": {"Age": {"mean": 29.7, "min": 0.42, "max": 80.0}}
}
```

## POST /api/v1/predict

LLM 없이 동작하는 ML 예측 엔드포인트. 기존 시스템이 예측 기능만 연계할 때 사용합니다.

**요청**

```json
{
  "pclass": 1,          // 필수, 1|2|3
  "sex": "female",      // 필수, male|female
  "age": 30,            // 필수, 0~120
  "sibsp": 0,           // 선택, 기본 0
  "parch": 0,           // 선택, 기본 0
  "fare": null,         // 선택, 생략 시 해당 등급 운임 중앙값으로 대치
  "embarked": "S"       // 선택, S|C|Q, 기본 S
}
```

**응답 200**

```json
{
  "survived": true,
  "survival_probability": 0.879,
  "model_cv_accuracy": 0.8272
}
```

- `survival_probability`: 보정된(calibrated) 생존 확률
- `model_cv_accuracy`: 학습 시점의 5-fold 교차검증 정확도 — 소비 시스템이
  예측 신뢰 수준을 함께 표시할 수 있도록 노출

**오류**: 유효성 위반(등급 5, 음수 나이 등)은 `422` + pydantic 상세 메시지

## POST /api/v1/chat

자연어 질문을 AI Agent가 처리합니다. 도구 호출 내역을 함께 반환해
소비 시스템이 "근거"를 표시할 수 있습니다.

**요청**

```json
{
  "message": "성별 생존율을 비교해줘",       // 필수, 1~2000자
  "session_id": null                        // 선택, 멀티턴 대화 이어가기
}
```

**응답 200**

```json
{
  "answer": "여성의 생존율이 남성보다 크게 높습니다. 여성 74.2%(314명 중 233명), 남성 18.9%(577명 중 109명)입니다.",
  "session_id": "3f2a...",
  "tool_calls": [
    {"name": "analyze_survival", "input": {"group_by": "Sex"}, "is_error": false}
  ],
  "stop_reason": "end_turn"
}
```

| 필드 | 설명 |
|---|---|
| `session_id` | 응답의 값을 다음 요청에 넣으면 대화가 이어짐 (서버 인메모리, 최대 100세션 FIFO) |
| `tool_calls` | 에이전트가 실행한 도구와 입력 — 감사/디버깅/근거 표시용 |
| `stop_reason` | `end_turn`(정상) / `max_turns_exceeded`(반복 상한 도달) / `refusal`(정책상 거부) |

**오류**

| 코드 | 조건 | 대응 |
|---|---|---|
| 503 | LLM 미설정 | `ANTHROPIC_API_KEY` 설정 후 재기동, 또는 LLM 비의존 엔드포인트만 사용 |
| 404 | 알 수 없는 `session_id` | 새 세션으로 시작 (서버 재시작·FIFO 정리로 세션 소멸 가능) |
| 422 | 빈 메시지, 2000자 초과 | 요청 보정 |

## 연계 시나리오 예시

**시나리오 A — 사내 챗봇에 분석 기능 추가**

```
사용자 → 챗봇 UI → POST /chat (session_id 유지) → answer + tool_calls 렌더링
```

**시나리오 B — 기존 백오피스에 예측 배지 표시**

```
백오피스 서버 → POST /predict → survival_probability를 화면에 표시 (LLM 계약 불필요)
```

**시나리오 C — 모니터링**

```
로드밸런서 → GET /health → llm_available=false 시 챗 기능만 숨김 (부분 장애 격리)
```

## 운영 시 고려 사항 (현재 범위 밖, 전환 지점 명시)

- **인증**: 데모에는 없음 → 게이트웨이 레벨 API Key/JWT 추가 지점은 FastAPI 미들웨어
- **세션 저장소**: 인메모리 → Redis 교체 시 `api.py`의 `sessions` dict만 대체
- **레이트 리밋**: LLM 비용 보호를 위해 `/chat`에 우선 적용 권장
