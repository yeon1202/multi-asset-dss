"""
설정 파일(config/*.yaml) 로더.

PROJECT_SPEC.md 원칙: "하드코딩 금지". 따라서 모든 임계값/자산 리스트는
이 모듈을 통해 YAML에서 읽어옵니다.
"""
from __future__ import annotations

from functools import lru_cache  # 같은 호출 결과를 메모리에 저장(메모이제이션)
from pathlib import Path
from typing import Any

import yaml  # PyYAML — YAML 파싱 라이브러리

# 프로젝트 루트 = 이 파일에서 ../../ 두 단계 위
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


@lru_cache(maxsize=None)  # 데코레이터 — 같은 인자에 대해 단 한 번만 디스크 읽기
def load_yaml(filename: str) -> dict[str, Any]:
    """config/<filename> 을 dict로 반환."""
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {path}")
    # with 문: 파일을 열고 작업 후 자동으로 닫아주는 구문
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{filename} 의 최상위는 dict 이어야 합니다.")
    return data


def load_universe() -> dict[str, Any]:
    """자산 유니버스 설정."""
    return load_yaml("universe.yaml")


def load_thresholds() -> dict[str, Any]:
    """기술적 지표 임계값 설정."""
    return load_yaml("thresholds.yaml")


def load_fundamental_config() -> dict[str, Any]:
    """Phase 2 — 펀더멘털 스코어링 설정."""
    return load_yaml("fundamental.yaml")


def load_macro_config() -> dict[str, Any]:
    """Phase 3 — 거시·레짐 설정."""
    return load_yaml("macro.yaml")


def load_portfolio_config() -> dict[str, Any]:
    """Phase 4 — 포트폴리오 최적화 설정."""
    return load_yaml("portfolio.yaml")
