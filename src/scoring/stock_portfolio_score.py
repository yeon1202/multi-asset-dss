"""
한국 주식 전용 통합 점수 — Phase 8.

Phase 1 (기술적 모멘텀) + Phase 2 (펀더멘털) + Phase 3 (레짐 적합도) 세 신호 결합.
각 종목당 0-100 점수 산출. ETF 트랙(Phase 4)과 별도.

세 신호의 정신:
  - 기술적 모멘텀: "최근 3개월 주가 추세" — 단기 신호
  - 펀더멘털: "PER/PBR/ROE 등 가치·수익성" — 중기 신호
  - 레짐 적합도: "지금 시장에서 이 섹터가 어떨까" — 거시 신호
"""
from __future__ import annotations

from typing import Mapping

import pandas as pd

from src.scoring.composite_score import technical_score
from src.scoring.regime_fit import asset_fit_score


def sector_regime_scores(
    sector_map: Mapping[str, str],
    sector_preferences: Mapping[str, float],
    default_preference: float,
    regime_score: float,
) -> pd.Series:
    """
    종목별 섹터에서 선호도 가져와 레짐 적합도(0-100) 계산.

    Parameters
    ----------
    sector_map : {종목코드: 섹터명}
    sector_preferences : {섹터명: 선호도(-1~+1)}
    default_preference : 섹터 매핑에 없으면 사용
    regime_score : 현재 레짐 점수 (-1~+1)
    """
    out = {}
    for code, sector in sector_map.items():
        pref = sector_preferences.get(sector, default_preference)
        out[code] = asset_fit_score(pref, regime_score)
    return pd.Series(out)


def stock_composite_score(
    prices: pd.DataFrame,                      # columns = 종목코드
    fundamental_scores: pd.Series,             # index = 종목코드, 0-100
    regime_score: float,                        # -1 ~ +1
    sector_map: Mapping[str, str],             # {종목코드: 섹터명}
    sector_preferences: Mapping[str, float],
    default_preference: float,
    weights: Mapping[str, float],              # {technical, fundamental, regime}
    technical_cfg: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """
    종목별 4컬럼 점수 DataFrame.

    Returns
    -------
    pd.DataFrame
        index = 종목코드,
        columns = [technical, fundamental, regime, composite] — 모두 0-100
    """
    tech_cfg = dict(technical_cfg or {})
    tech = technical_score(
        prices,
        lookback_days=int(tech_cfg.get("momentum_lookback_days", 63)),
        use_vol_adjusted=bool(tech_cfg.get("use_vol_adjusted", True)),
    )

    regime = sector_regime_scores(
        sector_map, sector_preferences, default_preference, regime_score
    )

    # 세 시리즈의 공통 종목만
    common = tech.index.intersection(fundamental_scores.index).intersection(regime.index)
    if len(common) == 0:
        raise ValueError("기술적·펀더멘털·레짐 점수 사이에 공통 종목이 없음")

    tech = tech.loc[common]
    fund = fundamental_scores.loc[common]
    regime = regime.loc[common]

    w_t = float(weights.get("technical", 0.33))
    w_f = float(weights.get("fundamental", 0.34))
    w_r = float(weights.get("regime", 0.33))
    total = w_t + w_f + w_r
    if total <= 0:
        raise ValueError("가중치 합이 0 이하")
    w_t, w_f, w_r = w_t / total, w_f / total, w_r / total

    # 펀더멘털이 NaN 인 종목 (금융주의 OPM/Debt 결측 등) 은
    # 가중치 재정규화로 처리 — 두 신호만으로 점수 산출
    rows = []
    for code in common:
        t, f, r = tech.get(code), fund.get(code), regime.get(code)
        contribs = []
        if pd.notna(t):
            contribs.append((t, w_t))
        if pd.notna(f):
            contribs.append((f, w_f))
        if pd.notna(r):
            contribs.append((r, w_r))
        if not contribs:
            comp = float("nan")
        else:
            wsum = sum(w for _, w in contribs)
            comp = sum(v * w for v, w in contribs) / wsum
        rows.append({
            "code": code,
            "technical": t,
            "fundamental": f,
            "regime": r,
            "composite": comp,
        })
    return pd.DataFrame(rows).set_index("code")
