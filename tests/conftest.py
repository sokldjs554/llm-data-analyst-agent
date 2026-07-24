"""공용 픽스처. src 레이아웃 패키지를 설치 없이 임포트할 수 있게 경로를 추가한다."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from titanic_agent.data import load_dataset  # noqa: E402
from titanic_agent.ml import SurvivalModel  # noqa: E402
from titanic_agent.tools import ToolExecutor  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "titanic.csv"


@pytest.fixture(scope="session")
def df():
    return load_dataset(DATA_PATH)


@pytest.fixture(scope="session")
def model(df):
    m = SurvivalModel()
    m.train(df)
    return m


@pytest.fixture(scope="session")
def executor(df, model):
    return ToolExecutor(df, model)
