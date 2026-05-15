"""
시장 국면(레짐) 분류기 — Phase 3.

알고리즘 (룰 기반, 블랙박스 X):
  1. 각 거시 feature 의 현재 값을 [-1, +1] 점수로 정규화
     - direction=+1: 값↑ → +1 (risk-on)
     - direction=-1: 값↑ → -1 (risk-off)
     - lo 이하 또는 hi 이상이면 포화. 사이는 선형 보간.
  2. 가중 평균 → 종합 점수 ∈ [-1, +1]
  3. 임계값으로 risk_on / neutral / risk_off 라벨

설명 가능성:
  - classify_with_explanation() 은 각 feature 의 기여도를 함께 반환.
  - PROJECT_SPEC.md §9.3 "근거 표시 필수" 원칙.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeResult:
    """단일 시점의 레짐 판정 결과."""

    label: str                       # "risk_on" | "neutral" | "risk_off"
    score: float                     # [-1, +1]
    contributions: dict[str, float]  # feature_name -> 가중치 * 정규화점수
    feature_scores: dict[str, float]  # feature_name -> 정규화점수 (-1~+1)
    feature_values: dict[str, float]  # feature_name -> 원본 값


def feature_score(value: float, lo: float, hi: float, direction: int) -> float:
    """
    한 feature 의 값을 [-1, +1] 점수로 정규화.

    direction=-1 (값이 클수록 risk-off, 예: VIX):
      value ≤ lo  → +1 (risk-on)
      value ≥ hi  → -1 (risk-off)
      그 사이      → 선형 보간

    direction=+1 (값이 클수록 risk-on, 예: 장단기 스프레드):
      value ≤ lo  → -1
      value ≥ hi  → +1
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    if direction not in (-1, 1):
        raise ValueError("direction 은 -1 또는 +1")
    if lo >= hi:
        raise ValueError("lo < hi 필요")

    # 0~1 정규화 (lo→0, hi→1)
    t = (float(value) - lo) / (hi - lo)
    t = max(0.0, min(1.0, t))
    # 0~1 → -1~+1 (lo→-1, hi→+1)
    raw = 2 * t - 1
    # direction=+1: 값↑ → risk-on(+1) — 그대로 반환
    # direction=-1: 값↑ → risk-off(-1) — 부호 반전
    return raw * direction


def _normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("가중치 합이 0 이하")
    return {k: v / total for k, v in weights.items()}


def classify_regime(
    feature_values: Mapping[str, float],
    config: dict,
) -> RegimeResult:
    """
    한 시점의 거시 feature 값들을 받아 레짐을 판정.

    Parameters
    ----------
    feature_values : dict
        {"vix": 18.5, "hy_oas": 3.4, ...}
    config : dict
        load_macro_config()["regime"] 의 내용.

    Returns
    -------
    RegimeResult
    """
    rules = config["features"]
    weights_raw = {name: rules[name]["weight"] for name in rules}

    # NaN 인 feature 는 가중치 재정규화로 제외
    feature_scores: dict[str, float] = {}
    valid_weights: dict[str, float] = {}
    for name, rule in rules.items():
        val = feature_values.get(name, float("nan"))
        if val is None or (isinstance(val, float) and math.isnan(val)):
            feature_scores[name] = float("nan")
            continue
        s = feature_score(val, rule["lo"], rule["hi"], rule["direction"])
        feature_scores[name] = s
        valid_weights[name] = weights_raw[name]

    if not valid_weights:
        return RegimeResult(
            label="neutral", score=float("nan"),
            contributions={}, feature_scores=feature_scores,
            feature_values=dict(feature_values),
        )

    norm_w = _normalize_weights(valid_weights)
    contributions = {name: feature_scores[name] * norm_w[name] for name in norm_w}
    score = sum(contributions.values())

    th = config["thresholds"]
    if score >= th["risk_on"]:
        label = "risk_on"
    elif score <= th["risk_off"]:
        label = "risk_off"
    else:
        label = "neutral"

    return RegimeResult(
        label=label, score=score,
        contributions=contributions,
        feature_scores=feature_scores,
        feature_values=dict(feature_values),
    )


def detect_history(
    macro_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    macro_df 의 각 행(날짜)에 대해 레짐을 판정 → 일별 히스토리.

    Parameters
    ----------
    macro_df : DataFrame
        index = DatetimeIndex,
        columns = config["features"] 의 키 (vix, hy_oas, ...)

    Returns
    -------
    pd.DataFrame
        index = 날짜,
        columns = [score, label] + 각 feature 의 정규화점수.
    """
    if macro_df.empty:
        return pd.DataFrame(columns=["score", "label"])

    rows: list[dict] = []
    feature_names = list(config["features"].keys())
    for ts, row in macro_df.iterrows():
        values = {name: row.get(name) for name in feature_names}
        r = classify_regime(values, config)
        out = {"score": r.score, "label": r.label}
        for name in feature_names:
            out[f"{name}_score"] = r.feature_scores.get(name, float("nan"))
        rows.append(out)
    return pd.DataFrame(rows, index=macro_df.index)
