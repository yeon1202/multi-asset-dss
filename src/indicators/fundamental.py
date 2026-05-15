"""
펀더멘털 지표 — Phase 2.

5가지 비율을 계산:
  - PER (Price-to-Earnings)   = 시가총액 / 당기순이익
  - PBR (Price-to-Book)       = 시가총액 / 자본총계
  - ROE (Return on Equity)    = 당기순이익 / 자본총계
  - OPM (Operating Margin)    = 영업이익 / 매출액
  - Debt (부채비율)           = 부채총계 / 자본총계

모든 함수는 분모 0/음수, 결측치(NaN) 입력에 대해 NaN 을 반환.
PROJECT_SPEC.md §7.3 결측치 처리 원칙.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _safe_divide(numerator: float, denominator: float) -> float:
    """0 또는 음수 분모면 NaN. 그 외엔 정상 나눗셈."""
    if denominator is None or numerator is None:
        return math.nan
    try:
        n = float(numerator)
        d = float(denominator)
    except (TypeError, ValueError):
        return math.nan
    if math.isnan(n) or math.isnan(d):
        return math.nan
    if d <= 0:  # 음수·0 분모는 비율 해석 불가
        return math.nan
    return n / d


def per(market_cap: float, net_income: float) -> float:
    """시가총액 / 당기순이익. 적자(순이익 ≤ 0) 면 NaN."""
    return _safe_divide(market_cap, net_income)


def pbr(market_cap: float, total_equity: float) -> float:
    """시가총액 / 자본총계. 자본잠식 시 NaN."""
    return _safe_divide(market_cap, total_equity)


def roe(net_income: float, total_equity: float) -> float:
    """당기순이익 / 자본총계. 자본잠식 시 NaN. 적자는 음수 그대로."""
    if total_equity is None or total_equity <= 0:
        return math.nan
    if net_income is None or (isinstance(net_income, float) and math.isnan(net_income)):
        return math.nan
    return float(net_income) / float(total_equity)


def opm(op_profit: float, revenue: float) -> float:
    """영업이익률 = 영업이익 / 매출액. 매출 ≤ 0 면 NaN."""
    if revenue is None or revenue <= 0:
        return math.nan
    if op_profit is None or (isinstance(op_profit, float) and math.isnan(op_profit)):
        return math.nan
    return float(op_profit) / float(revenue)


def debt_ratio(total_debt: float, total_equity: float) -> float:
    """부채비율 = 부채총계 / 자본총계."""
    return _safe_divide(total_debt, total_equity)


def compute_ratios(
    market_cap: float,
    revenue: float,
    op_profit: float,
    net_income: float,
    total_assets: float,
    total_debt: float,
    total_equity: float,
) -> dict[str, float]:
    """다섯 비율을 한 번에 계산해 dict 로 반환."""
    return {
        "per": per(market_cap, net_income),
        "pbr": pbr(market_cap, total_equity),
        "roe": roe(net_income, total_equity),
        "opm": opm(op_profit, revenue),
        "debt": debt_ratio(total_debt, total_equity),
    }


def compute_ratios_table(
    financials: pd.DataFrame,
    market_caps: pd.DataFrame,
) -> pd.DataFrame:
    """
    여러 종목을 한 번에. financials 와 market_caps 의 인덱스(code)를 기준으로 조인.

    Parameters
    ----------
    financials : DataFrame
        index = code, columns 일부: revenue, op_profit, net_income,
                                     total_assets, total_debt, total_equity
    market_caps : DataFrame
        index = code, columns 일부: market_cap

    Returns
    -------
    pd.DataFrame
        index = code, columns = [per, pbr, roe, opm, debt]
    """
    # 외부 조인: 한쪽이라도 결측이면 NaN 으로 둠
    merged = financials.join(market_caps[["market_cap"]], how="left")

    needed = [
        "market_cap", "revenue", "op_profit", "net_income",
        "total_assets", "total_debt", "total_equity",
    ]
    for col in needed:
        if col not in merged.columns:
            merged[col] = np.nan

    rows = []
    for code, r in merged.iterrows():
        ratios = compute_ratios(
            market_cap=r["market_cap"],
            revenue=r["revenue"],
            op_profit=r["op_profit"],
            net_income=r["net_income"],
            total_assets=r["total_assets"],
            total_debt=r["total_debt"],
            total_equity=r["total_equity"],
        )
        ratios["code"] = code
        rows.append(ratios)

    out = pd.DataFrame(rows).set_index("code")
    return out[["per", "pbr", "roe", "opm", "debt"]]
