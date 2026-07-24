"""에이전트 도구 정의 및 실행기.

LLM에게 임의 코드 실행 권한을 주는 대신, 구조화된 질의 인터페이스만 노출한다.
- 허용된 컬럼/연산만 사용 가능 → 프롬프트 인젝션으로 인한 임의 코드 실행 차단
- 모든 도구는 JSON 문자열을 반환 → LLM이 수치를 그대로 인용 가능
설계 근거는 docs/02-agent-strategy.md 참고.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .data import CATEGORICAL_COLUMNS, QUERYABLE_COLUMNS, dataset_overview
from .ml import SurvivalModel

ALLOWED_OPS = {"==", "!=", ">", ">=", "<", "<="}

# Anthropic Messages API 형식의 도구 스키마
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "get_dataset_overview",
        "description": (
            "타이타닉 데이터셋의 요약 정보를 반환한다. "
            "행/열 수, 컬럼 스키마, 결측치 현황, 전체 생존율, 주요 수치 통계를 포함한다. "
            "데이터셋에 대한 일반적인 질문이나 분석을 시작할 때 먼저 호출하라."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "analyze_survival",
        "description": (
            "지정한 컬럼 기준으로 그룹별 승객 수, 생존자 수, 생존율을 계산한다. "
            "'성별 생존율', '객실 등급별 생존율' 같은 질문에 사용하라. "
            "filters로 특정 조건의 승객만 대상으로 분석할 수 있다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "group_by": {
                    "type": "string",
                    "description": f"그룹 기준 컬럼. 허용값: {sorted(CATEGORICAL_COLUMNS - {'Survived'})}",
                },
                "filters": {
                    "type": "array",
                    "description": "선택. 분석 대상을 좁히는 조건 목록 (AND 결합)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string"},
                            "op": {"type": "string", "enum": sorted(ALLOWED_OPS)},
                            "value": {},
                        },
                        "required": ["column", "op", "value"],
                    },
                },
            },
            "required": ["group_by"],
        },
    },
    {
        "name": "get_statistics",
        "description": (
            "특정 컬럼의 통계를 반환한다. 수치형 컬럼은 기술통계(평균/중앙값/사분위수), "
            "범주형 컬럼은 값별 빈도를 반환한다. filters로 조건부 통계도 가능하다. "
            "예: '1등석 승객의 평균 나이', '탑승 항구별 인원'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {
                    "type": "string",
                    "description": f"대상 컬럼. 허용값: {[c for c in QUERYABLE_COLUMNS if c not in ('PassengerId', 'Name')]}",
                },
                "filters": {
                    "type": "array",
                    "description": "선택. 조건 목록 (AND 결합)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string"},
                            "op": {"type": "string", "enum": sorted(ALLOWED_OPS)},
                            "value": {},
                        },
                        "required": ["column", "op", "value"],
                    },
                },
            },
            "required": ["column"],
        },
    },
    {
        "name": "predict_survival",
        "description": (
            "학습된 ML 모델(SVC)로 가상 승객의 생존 확률을 예측한다. "
            "'30세 여성이 1등석에 탔다면 생존했을까?' 같은 가정 질문에 사용하라."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pclass": {"type": "integer", "enum": [1, 2, 3], "description": "객실 등급"},
                "sex": {"type": "string", "enum": ["male", "female"], "description": "성별"},
                "age": {"type": "number", "description": "나이"},
                "sibsp": {"type": "integer", "description": "동반한 형제/배우자 수 (기본 0)"},
                "parch": {"type": "integer", "description": "동반한 부모/자녀 수 (기본 0)"},
                "fare": {"type": "number", "description": "운임. 생략 시 중앙값으로 대치"},
                "embarked": {
                    "type": "string",
                    "enum": ["S", "C", "Q"],
                    "description": "탑승 항구 (기본 S)",
                },
            },
            "required": ["pclass", "sex", "age"],
        },
    },
]


class ToolError(Exception):
    """도구 실행 중 사용자에게 전달 가능한 오류."""


class ToolExecutor:
    """도구 이름과 입력을 받아 실제 데이터/모델 연산을 수행한다."""

    def __init__(self, df: pd.DataFrame, model: SurvivalModel) -> None:
        self.df = df
        self.model = model

    # ── 공개 인터페이스 ────────────────────────────────────────────────
    def execute(self, name: str, tool_input: dict[str, Any]) -> str:
        """도구를 실행하고 결과를 JSON 문자열로 반환한다.

        오류는 ToolError로 던지며, 호출 측(agent)은 이를 is_error 도구 결과로
        변환해 LLM이 스스로 정정할 수 있게 한다.
        """
        handlers = {
            "get_dataset_overview": self._overview,
            "analyze_survival": self._analyze_survival,
            "get_statistics": self._statistics,
            "predict_survival": self._predict,
        }
        handler = handlers.get(name)
        if handler is None:
            raise ToolError(f"알 수 없는 도구입니다: {name}")
        try:
            result = handler(**tool_input)
        except ToolError:
            raise
        except (TypeError, ValueError, KeyError) as exc:
            raise ToolError(f"도구 입력이 올바르지 않습니다: {exc}") from exc
        return json.dumps(result, ensure_ascii=False, default=str)

    # ── 내부 구현 ─────────────────────────────────────────────────────
    def _overview(self) -> dict:
        return dataset_overview(self.df)

    def _apply_filters(self, df: pd.DataFrame, filters: list[dict] | None) -> pd.DataFrame:
        if not filters:
            return df
        for f in filters:
            column, op, value = f.get("column"), f.get("op"), f.get("value")
            if column not in QUERYABLE_COLUMNS:
                raise ToolError(
                    f"허용되지 않은 컬럼입니다: {column}. 사용 가능: {QUERYABLE_COLUMNS}"
                )
            if op not in ALLOWED_OPS:
                raise ToolError(f"허용되지 않은 연산자입니다: {op}. 사용 가능: {sorted(ALLOWED_OPS)}")
            series = df[column]
            # 문자열로 들어온 수치 값 보정 (LLM 입력 방어)
            if pd.api.types.is_numeric_dtype(series) and isinstance(value, str):
                try:
                    value = float(value)
                except ValueError as exc:
                    raise ToolError(
                        f"수치형 컬럼 {column}에 수치가 아닌 값이 왔습니다: {value!r}"
                    ) from exc
            if op == "==":
                mask = series == value
            elif op == "!=":
                mask = series != value
            elif op == ">":
                mask = series > value
            elif op == ">=":
                mask = series >= value
            elif op == "<":
                mask = series < value
            else:
                mask = series <= value
            df = df[mask]
        return df

    def _analyze_survival(self, group_by: str, filters: list[dict] | None = None) -> dict:
        allowed = sorted(CATEGORICAL_COLUMNS - {"Survived"})
        if group_by not in allowed:
            raise ToolError(f"group_by는 다음 중 하나여야 합니다: {allowed}")
        df = self._apply_filters(self.df, filters)
        if df.empty:
            return {"group_by": group_by, "groups": [], "note": "조건에 맞는 승객이 없습니다."}
        grouped = df.groupby(group_by, observed=True)["Survived"].agg(["count", "sum", "mean"])
        return {
            "group_by": group_by,
            "total_passengers": int(len(df)),
            "groups": [
                {
                    "group": str(idx),
                    "passengers": int(row["count"]),
                    "survivors": int(row["sum"]),
                    "survival_rate": round(float(row["mean"]), 4),
                }
                for idx, row in grouped.iterrows()
            ],
        }

    def _statistics(self, column: str, filters: list[dict] | None = None) -> dict:
        if column not in QUERYABLE_COLUMNS or column in ("PassengerId", "Name"):
            raise ToolError(f"통계를 제공하지 않는 컬럼입니다: {column}")
        df = self._apply_filters(self.df, filters)
        series = df[column]
        if series.dropna().empty:
            return {"column": column, "note": "조건에 맞는 데이터가 없습니다."}
        if column in CATEGORICAL_COLUMNS or not pd.api.types.is_numeric_dtype(series):
            counts = series.value_counts(dropna=True)
            return {
                "column": column,
                "type": "categorical",
                "total": int(series.notna().sum()),
                "value_counts": {str(k): int(v) for k, v in counts.items()},
            }
        desc = series.describe()
        return {
            "column": column,
            "type": "numeric",
            "count": int(desc["count"]),
            "mean": round(float(desc["mean"]), 2),
            "std": round(float(desc["std"]), 2),
            "min": round(float(desc["min"]), 2),
            "25%": round(float(desc["25%"]), 2),
            "median": round(float(desc["50%"]), 2),
            "75%": round(float(desc["75%"]), 2),
            "max": round(float(desc["max"]), 2),
        }

    def _predict(
        self,
        pclass: int,
        sex: str,
        age: float,
        sibsp: int = 0,
        parch: int = 0,
        fare: float | None = None,
        embarked: str = "S",
    ) -> dict:
        if fare is None:
            # 같은 등급 승객의 운임 중앙값으로 대치
            fare = float(self.df.loc[self.df["Pclass"] == pclass, "Fare"].median())
        result = self.model.predict(
            pclass=pclass, sex=sex, age=age, sibsp=sibsp, parch=parch, fare=fare, embarked=embarked
        )
        return {
            "input": {
                "pclass": pclass, "sex": sex, "age": age,
                "sibsp": sibsp, "parch": parch, "fare": round(fare, 2), "embarked": embarked,
            },
            "survived": result.survived,
            "survival_probability": result.probability,
            "model": "SVC (5-fold CV accuracy: "
            + (f"{self.model.cv_accuracy}" if self.model.cv_accuracy else "미평가")
            + ")",
        }
