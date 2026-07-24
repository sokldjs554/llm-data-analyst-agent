"""생존 예측 ML 모델.

기존 노트북의 SVC 모델을 sklearn Pipeline으로 재구성했다.
노트북의 수동 전처리(fillna, 라벨 인코딩)를 ColumnTransformer로 대체해
학습과 추론에서 동일한 변환이 보장되도록 개선한 것이 핵심 차이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

NUMERIC_FEATURES = ["Age", "Fare", "SibSp", "Parch", "Familysize"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked"]
TARGET = "Survived"


@dataclass
class PredictionResult:
    survived: bool
    probability: float  # 생존 확률 (0~1)


def build_pipeline() -> Pipeline:
    """전처리 + SVC 분류기 파이프라인을 생성한다."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            # sklearn 1.9에서 SVC(probability=True)가 deprecated되어
            # 권장 방식인 CalibratedClassifierCV로 확률을 보정한다
            ("model", CalibratedClassifierCV(SVC(random_state=42), ensemble=False)),
        ]
    )


class SurvivalModel:
    """학습/평가/예측을 담당하는 생존 예측 모델 래퍼."""

    def __init__(self) -> None:
        self.pipeline = build_pipeline()
        self.cv_accuracy: float | None = None

    def train(self, df: pd.DataFrame) -> float:
        """모델을 학습하고 5-fold 교차검증 정확도를 반환한다."""
        features = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        target = df[TARGET]
        scores = cross_val_score(build_pipeline(), features, target, cv=5)
        self.cv_accuracy = round(float(scores.mean()), 4)
        self.pipeline.fit(features, target)
        return self.cv_accuracy

    def predict(
        self,
        pclass: int,
        sex: str,
        age: float,
        sibsp: int = 0,
        parch: int = 0,
        fare: float | None = None,
        embarked: str = "S",
    ) -> PredictionResult:
        """단일 승객 정보를 받아 생존 여부와 확률을 반환한다."""
        row = pd.DataFrame(
            [
                {
                    "Pclass": pclass,
                    "Sex": sex,
                    "Age": age,
                    "SibSp": sibsp,
                    "Parch": parch,
                    "Fare": fare,
                    "Embarked": embarked,
                    "Familysize": sibsp + parch,
                }
            ]
        )
        proba = float(self.pipeline.predict_proba(row)[0][1])
        return PredictionResult(survived=proba >= 0.5, probability=round(proba, 4))
