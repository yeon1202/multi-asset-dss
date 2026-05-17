"""
켈리 공식 — Phase 4.

Kelly fraction (단일 위험자산, 무위험금리 = r_f):
    f* = (μ - r_f) / σ²

Half-Kelly = f* / 2 — 실전에서는 추정오차 때문에 풀 켈리는 위험.
PROJECT_SPEC.md 도 하프 켈리를 명시.

다자산 포트폴리오에서는 마코위츠로 비중 w 를 먼저 정한 뒤,
포트폴리오 전체에 Kelly 스케일을 곱해 위험자산 vs 현금 비중을 결정.
"""
from __future__ import annotations

import math

import pandas as pd


def kelly_fraction(
    expected_return: float,
    volatility: float,
    risk_free_rate: float = 0.0,
) -> float:
    """
    단일 위험 포트폴리오의 Kelly 비율.

    f* = (μ - r_f) / σ²

    음수면 (기대수익 < 무위험) 0 으로. 음수 베팅(공매도)은 이 단계에서 사용 X.
    """
    if volatility is None or volatility <= 0:
        return 0.0
    if expected_return is None or math.isnan(expected_return):
        return 0.0
    f = (float(expected_return) - float(risk_free_rate)) / (float(volatility) ** 2)
    return max(0.0, f)


def half_kelly(
    expected_return: float,
    volatility: float,
    risk_free_rate: float = 0.0,
    fraction: float = 0.5,
    cap: float = 1.0,
) -> float:
    """
    스케일된 Kelly. fraction=0.5 가 Half-Kelly.

    cap 으로 위험자산 총 비중 상한도 같이 적용.
    """
    f = kelly_fraction(expected_return, volatility, risk_free_rate)
    return min(cap, max(0.0, fraction * f))


def apply_kelly_sizing(
    risky_weights: pd.Series,
    expected_return: float,
    volatility: float,
    risk_free_rate: float = 0.0,
    kelly_fraction_param: float = 0.5,
    max_total_risk_weight: float = 0.95,
    min_cash_weight: float = 0.05,
) -> pd.Series:
    """
    마코위츠 결과(risky_weights, sum=1)에 켈리 스케일을 곱해
    위험자산 비중 + 현금 비중 = 1 인 최종 배분을 만듦.

    Parameters
    ----------
    risky_weights : Series
        sum=1 의 마코위츠 결과 (자산코드 인덱스).
    expected_return, volatility : float
        해당 포트폴리오의 연환산 μ, σ.
    risk_free_rate : float
    kelly_fraction_param : float
        0.5 = Half-Kelly.
    max_total_risk_weight : float
        위험자산 합 상한 (예: 0.95 → 현금 최소 5%).
    min_cash_weight : float
        현금 최소 비중. max_total_risk_weight 와 일치시키기 위함.

    Returns
    -------
    pd.Series
        index = 자산코드 + "CASH", values = sum 1.
    """
    cap = min(max_total_risk_weight, 1.0 - min_cash_weight)
    k = half_kelly(
        expected_return, volatility, risk_free_rate,
        fraction=kelly_fraction_param, cap=cap,
    )
    scaled = (risky_weights * k).astype(float)
    cash = 1.0 - scaled.sum()
    # 부동소수점 오차 안전화
    if cash < min_cash_weight:
        # 위험자산을 줄여 현금 비중 맞춤
        excess = min_cash_weight - cash
        scaled = scaled * (1 - excess / scaled.sum()) if scaled.sum() > 0 else scaled
        cash = 1.0 - scaled.sum()
    result = scaled.copy()
    result["CASH"] = cash
    return result
