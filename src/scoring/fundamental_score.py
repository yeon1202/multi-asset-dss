"""
펀더멘털 점수 산출 — Phase 2.

방법:
  1. 각 지표를 sanity_bounds 로 winsorize (이상치 → NaN).
  2. 종목 간 백분위(percentile)로 [0, 100] 정규화.
     - "높을수록 좋음" (ROE, OPM): 그대로 백분위.
     - "낮을수록 좋음" (PER, PBR, Debt): 1 - 백분위.
  3. score_weights 로 가중 평균 → 0~100 종합 점수.

  ★ 어느 한 항목이 NaN 이면 나머지로 정규화하고 그 종목 가중치만 재정규화.

PROJECT_SPEC.md §9.3:
  - 단정 표현 금지. 점수는 "유니버스 내 상대 순위"임을 명시.
  - 모든 결과에 근거 표시 가능하도록 raw ratios 도 함께 반환.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# 점수 방향: True = 높을수록 좋음, False = 낮을수록 좋음
HIGHER_IS_BETTER: dict[str, bool] = {
    "per": False,
    "pbr": False,
    "roe": True,
    "opm": True,
    "debt": False,
}


def winsorize(ratios: pd.DataFrame, bounds: dict[str, dict[str, float]]) -> pd.DataFrame:
    """
    sanity_bounds 밖의 값은 NaN 으로 치환.

    예: PER=200 같은 비정상값은 점수 산출에서 제외 (적자/일회성 손익 등).
    """
    out = ratios.copy()
    for col, bound in bounds.items():
        if col not in out.columns:
            continue
        lo, hi = bound.get("min", -np.inf), bound.get("max", np.inf)
        mask = (out[col] < lo) | (out[col] > hi)
        out.loc[mask, col] = np.nan
    return out


def percentile_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    """
    한 컬럼을 [0, 100] 점수로 정규화 (NaN 은 NaN 으로 유지).

    pandas .rank(pct=True): 백분위(0~1). 동순위는 평균으로.
    """
    # method="average": 동값은 평균 순위. NaN 은 자동 제외.
    pct = series.rank(pct=True, method="average")
    score = pct * 100
    if not higher_is_better:
        # 낮을수록 좋음 → 뒤집기
        score = 100 - score
    return score


def fundamental_score(
    ratios: pd.DataFrame,
    weights: dict[str, float],
    sanity_bounds: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """
    종목별 펀더멘털 점수.

    Parameters
    ----------
    ratios : DataFrame
        index = code, columns 일부 또는 전부: per, pbr, roe, opm, debt.
    weights : dict
        예: {"per": 0.2, "pbr": 0.15, "roe": 0.3, "opm": 0.2, "debt": 0.15}.
        합이 1 이 아니어도 내부에서 정규화.
    sanity_bounds : dict, optional
        이상치 컷오프. None 이면 통과.

    Returns
    -------
    pd.DataFrame
        index = code,
        columns = [per, pbr, roe, opm, debt,                  ← 원본 (참고용)
                   per_score, pbr_score, roe_score, opm_score, debt_score,
                   composite_score]
        composite_score: 0~100. 결측 항목은 가중치 재정규화로 처리.
    """
    if sanity_bounds:
        clean = winsorize(ratios, sanity_bounds)
    else:
        clean = ratios.copy()

    score_cols: list[str] = []
    for col in HIGHER_IS_BETTER:
        if col not in clean.columns:
            continue
        s = percentile_score(clean[col], HIGHER_IS_BETTER[col])
        clean[f"{col}_score"] = s
        score_cols.append(f"{col}_score")

    # 종목별 합성 점수: NaN 인 항목은 빼고, 가중치 재정규화
    def _row_composite(row: pd.Series) -> float:
        used_w, total = 0.0, 0.0
        for col in HIGHER_IS_BETTER:
            sc = row.get(f"{col}_score")
            w = weights.get(col, 0.0)
            if sc is None or pd.isna(sc) or w == 0:
                continue
            total += sc * w
            used_w += w
        if used_w == 0:
            return float("nan")
        return total / used_w

    clean["composite_score"] = clean.apply(_row_composite, axis=1)

    # 원본 비율 + 항목 점수 + 합성 점수 순으로 컬럼 정리
    ordered: list[str] = []
    for col in HIGHER_IS_BETTER:
        if col in clean.columns:
            ordered.append(col)
    ordered += score_cols + ["composite_score"]
    return clean[ordered]


def top_n(scored: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """composite_score 내림차순 상위 N. NaN 은 제외."""
    if "composite_score" not in scored.columns:
        raise KeyError("scored DataFrame 에 composite_score 컬럼이 필요합니다.")
    return scored.dropna(subset=["composite_score"]).sort_values(
        "composite_score", ascending=False
    ).head(n)
