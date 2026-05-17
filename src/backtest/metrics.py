"""
백테스트 성과 지표 — Phase 5.

PROJECT_SPEC.md §6: 누적 수익률, 샤프지수, MDD, 승률.
+ CAGR, Calmar, Sortino 추가.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceMetrics:
    total_return: float       # 누적 수익률 (예: 0.85 = +85%)
    cagr: float               # 연환산 복리 수익률
    volatility: float         # 연환산 변동성
    sharpe: float             # (CAGR - r_f) / vol
    sortino: float            # 하방 변동성만 분모
    max_drawdown: float       # MDD (음수, 예: -0.30)
    calmar: float             # CAGR / |MDD|
    win_rate_monthly: float   # 월간 수익률 ≥ 0 비율
    n_obs: int                # 영업일 수


def total_return(equity: pd.Series) -> float:
    """누적 수익률."""
    if len(equity) < 2:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def cagr(equity: pd.Series) -> float:
    """연환산 복리 수익률."""
    if len(equity) < 2:
        return 0.0
    n_days = (equity.index[-1] - equity.index[0]).days
    if n_days <= 0:
        return 0.0
    years = n_days / 365.25
    ratio = equity.iloc[-1] / equity.iloc[0]
    if ratio <= 0:
        return float("nan")
    return float(ratio ** (1 / years) - 1)


def annualized_volatility(equity: pd.Series, factor: int = 252) -> float:
    """일간 수익률 표준편차 × √factor."""
    ret = equity.pct_change().dropna()
    if len(ret) < 2:
        return 0.0
    return float(ret.std(ddof=1) * np.sqrt(factor))


def sharpe_ratio(equity: pd.Series, risk_free_rate: float = 0.0) -> float:
    """(CAGR - r_f) / 연환산 변동성."""
    vol = annualized_volatility(equity)
    if vol <= 0:
        return 0.0
    return (cagr(equity) - risk_free_rate) / vol


def sortino_ratio(equity: pd.Series, risk_free_rate: float = 0.0,
                   factor: int = 252) -> float:
    """하방 변동성 기반 샤프."""
    ret = equity.pct_change().dropna()
    if len(ret) < 2:
        return 0.0
    downside = ret[ret < 0]
    if len(downside) < 2:
        return float("inf")
    downside_vol = downside.std(ddof=1) * np.sqrt(factor)
    if downside_vol <= 0:
        return 0.0
    return (cagr(equity) - risk_free_rate) / downside_vol


def max_drawdown(equity: pd.Series) -> float:
    """누적 최고치 대비 가장 큰 낙폭 (음수)."""
    if len(equity) < 2:
        return 0.0
    cummax = equity.cummax()
    drawdown = equity / cummax - 1
    return float(drawdown.min())


def calmar_ratio(equity: pd.Series) -> float:
    """CAGR / |MDD|."""
    mdd = abs(max_drawdown(equity))
    if mdd <= 0:
        return 0.0
    return cagr(equity) / mdd


def monthly_win_rate(equity: pd.Series) -> float:
    """월별 마지막 영업일 가치 차이가 양수인 비율."""
    monthly = equity.resample("ME").last().dropna()
    if len(monthly) < 2:
        return 0.0
    ret = monthly.pct_change().dropna()
    if len(ret) == 0:
        return 0.0
    return float((ret > 0).sum() / len(ret))


def compute_all(
    equity: pd.Series,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """주요 지표 한 묶음."""
    return PerformanceMetrics(
        total_return=total_return(equity),
        cagr=cagr(equity),
        volatility=annualized_volatility(equity),
        sharpe=sharpe_ratio(equity, risk_free_rate),
        sortino=sortino_ratio(equity, risk_free_rate),
        max_drawdown=max_drawdown(equity),
        calmar=calmar_ratio(equity),
        win_rate_monthly=monthly_win_rate(equity),
        n_obs=len(equity),
    )
