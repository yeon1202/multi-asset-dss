"""
다자산 통합 점수 — Phase 4.

Phase 1 (기술적 모멘텀) + Phase 3 (레짐 적합도) 를 결합해 자산별 0-100 점수.

Phase 2 (펀더멘털) 는 종목 선정용이라 다자산 최적화에는 직접 안 씀.
(Phase 1 의 KODEX 200 ETF 가 KOSPI 200 전체를 대표함)
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd


def momentum_returns(prices: pd.DataFrame, lookback_days: int) -> pd.Series:
    """
    각 자산의 lookback_days 일 전 대비 수익률.

    Parameters
    ----------
    prices : DataFrame
        columns = 자산코드, index = Date, values = 종가.
    lookback_days : int
        뒤로 몇 영업일.

    Returns
    -------
    pd.Series
        index = 자산코드, value = 누적 수익률 (예: 0.05 = +5%).
    """
    if len(prices) <= lookback_days:
        # 데이터 부족 시 0 으로 (점수화는 가능하도록)
        return pd.Series(0.0, index=prices.columns)
    end = prices.iloc[-1]
    start = prices.iloc[-(lookback_days + 1)]
    ret = (end / start) - 1
    return ret.astype(float)


def vol_adjusted_momentum(
    prices: pd.DataFrame,
    lookback_days: int,
    annualize_factor: int = 252,
) -> pd.Series:
    """
    변동성 조정 모멘텀 = 모멘텀 수익률 / (연환산 변동성).

    위험 대비 수익(샤프와 비슷한 정신)을 자산 간 비교용으로 사용.
    """
    ret = momentum_returns(prices, lookback_days)
    # 일간 수익률의 표준편차 × √(연환산)
    daily_ret = prices.pct_change()
    vol = daily_ret.tail(lookback_days).std(ddof=1) * np.sqrt(annualize_factor)
    # 변동성 0 또는 NaN 인 경우 → 단순 모멘텀
    safe = vol.where(vol > 0, np.nan)
    adjusted = ret / safe
    # NaN 인 곳은 단순 모멘텀으로 대체
    return adjusted.fillna(ret).astype(float)


def technical_score(
    prices: pd.DataFrame,
    lookback_days: int = 63,
    use_vol_adjusted: bool = True,
) -> pd.Series:
    """
    기술적 모멘텀을 0-100 백분위로 정규화.
    """
    if use_vol_adjusted:
        raw = vol_adjusted_momentum(prices, lookback_days)
    else:
        raw = momentum_returns(prices, lookback_days)
    # 자산이 1개뿐이면 50점 고정 (백분위 의미 없음)
    if len(raw) <= 1:
        return pd.Series(50.0, index=raw.index)
    return (raw.rank(pct=True) * 100).fillna(50.0)


def regime_fit_score(
    regime_score: float,
    preferences: Mapping[str, float],
) -> pd.Series:
    """
    Phase 3 의 자산 적합도를 Series 로 반환.

    fit = 50 + 50 × pref × regime_score
    """
    from src.scoring.regime_fit import asset_fit_score
    return pd.Series(
        {code: asset_fit_score(pref, regime_score) for code, pref in preferences.items()}
    )


def composite_score(
    prices: pd.DataFrame,
    regime_score: float,
    preferences: Mapping[str, float],
    weights: Mapping[str, float],
    technical_cfg: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """
    자산별 통합 점수.

    Returns
    -------
    pd.DataFrame
        index = 자산코드,
        columns = [technical, regime, composite] (모두 0-100).
    """
    tech_cfg = dict(technical_cfg or {})
    tech = technical_score(
        prices,
        lookback_days=int(tech_cfg.get("momentum_lookback_days", 63)),
        use_vol_adjusted=bool(tech_cfg.get("use_vol_adjusted", True)),
    )
    regime = regime_fit_score(regime_score, preferences)

    # 두 점수 정렬 (자산코드 공통)
    common = tech.index.intersection(regime.index)
    if len(common) == 0:
        raise ValueError("기술적 점수와 레짐 점수 사이에 공통 자산이 없음")
    tech = tech.loc[common]
    regime = regime.loc[common]

    w_tech = float(weights.get("technical", 0.5))
    w_regime = float(weights.get("regime", 0.5))
    total = w_tech + w_regime
    if total <= 0:
        raise ValueError("가중치 합이 0 이하")
    w_tech, w_regime = w_tech / total, w_regime / total

    comp = w_tech * tech + w_regime * regime
    out = pd.DataFrame({
        "technical": tech,
        "regime": regime,
        "composite": comp,
    })
    return out
