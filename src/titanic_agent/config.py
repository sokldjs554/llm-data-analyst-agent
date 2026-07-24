"""환경 변수 기반 설정."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# 프로젝트 루트 (src/titanic_agent/config.py 기준 두 단계 상위)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = "claude-opus-4-8"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "titanic.csv"


@dataclass
class Settings:
    """애플리케이션 설정. 환경 변수를 읽어 기본값을 채운다."""

    anthropic_api_key: str | None = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY") or None
    )
    model: str = field(
        default_factory=lambda: os.environ.get("TITANIC_AGENT_MODEL", DEFAULT_MODEL)
    )
    data_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("TITANIC_DATA_PATH", str(DEFAULT_DATA_PATH))
        )
    )
    max_agent_turns: int = 10
    max_tokens: int = 16000

    @property
    def llm_available(self) -> bool:
        return bool(self.anthropic_api_key)
