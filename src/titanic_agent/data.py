"""타이타닉 데이터셋 로딩 및 피처 엔지니어링.

기존 노트북(notebooks/titanic-eda.ipynb)의 전처리 로직을 재사용 가능한 모듈로 정리했다.
- Familysize = SibSp + Parch (노트북과 동일)
- 결측치 대치는 노트북의 수동 fillna 대신 ML 파이프라인(ml.py)에서 처리해
  학습/추론 시점의 일관성을 보장한다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# 에이전트 도구에서 조회를 허용하는 컬럼 (원본 + 파생)
QUERYABLE_COLUMNS = [
    "PassengerId",
    "Survived",
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "Familysize",
    "AgeGroup",
]

CATEGORICAL_COLUMNS = {"Survived", "Pclass", "Sex", "Embarked", "AgeGroup"}

# 노트북의 연령 구간(0-16-26-36-62-100)을 라벨과 함께 사용
AGE_BINS = [0, 16, 26, 36, 62, 100]
AGE_LABELS = ["child(0-16)", "young(17-26)", "adult(27-36)", "middle(37-62)", "senior(63+)"]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """CSV를 읽고 파생 피처를 추가해 반환한다."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"타이타닉 데이터셋을 찾을 수 없습니다: {path}")
    df = pd.read_csv(path)
    return add_derived_features(df)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """분석용 파생 컬럼을 추가한다 (원본은 변경하지 않음)."""
    out = df.copy()
    out["Familysize"] = out["SibSp"] + out["Parch"]
    out["AgeGroup"] = pd.cut(out["Age"], bins=AGE_BINS, labels=AGE_LABELS)
    return out


def dataset_overview(df: pd.DataFrame) -> dict:
    """데이터셋 요약 정보(스키마, 결측치, 기본 통계)를 dict로 반환한다."""
    numeric = df.select_dtypes("number")
    return {
        "num_rows": int(len(df)),
        "num_columns": int(len(df.columns)),
        "columns": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {
            col: int(cnt) for col, cnt in df.isna().sum().items() if cnt > 0
        },
        "survival_rate": round(float(df["Survived"].mean()), 4),
        "numeric_summary": {
            col: {
                "mean": round(float(numeric[col].mean()), 2),
                "min": round(float(numeric[col].min()), 2),
                "max": round(float(numeric[col].max()), 2),
            }
            for col in ("Age", "Fare", "Familysize")
            if col in numeric
        },
    }
