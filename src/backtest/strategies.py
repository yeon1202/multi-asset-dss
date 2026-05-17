"""
백테스트 전략 함수 — Phase 5.

각 전략은 시그니처 `(prices_up_to_date) -> weights` 를 따름.
prices 의 마지막 행이 "오늘" — look-ahead bias 방지.
PROJECT_SPEC.md §7.2 시점 정합성.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


# Strategy = Callable[[pd.DataFrame], pd.Series]
# Input: prices (index=Date, columns=자산코드), 마지막 행 = 그 시점.
# Output: 비중 Series (index=자산코드 + 옵션 "CASH"), 합 ≤ 1.


def buy_and_hold(asset_code: str) -> Callable[[pd.DataFrame], pd.Series]:
    """한 자산에만 100% 투자. 재조정 시점에도 그대로."""
    def strategy(prices: pd.DataFrame) -> pd.Series:
        cols = list(prices.columns)
        if asset_code not in cols:
            # 자산이 없으면 현금
            return pd.Series({"CASH": 1.0})
        w = {c: 0.0 for c in cols}
        w[asset_code] = 1.0
        return pd.Series(w)
    return strategy


def equal_weight() -> Callable[[pd.DataFrame], pd.Series]:
    """모든 자산을 동일 가중."""
    def strategy(prices: pd.DataFrame) -> pd.Series:
        n = len(prices.columns)
        if n == 0:
            return pd.Series({"CASH": 1.0})
        return pd.Series(1.0 / n, index=prices.columns)
    return strategy


def momentum_top_n(
    lookback_days: int = 63,
    top_n: int = 3,
    cash_when_no_momentum: bool = True,
) -> Callable[[pd.DataFrame], pd.Series]:
    """
    모멘텀 상위 N 자산을 동일 가중.

    Parameters
    ----------
    lookback_days : 모멘텀 측정 기간 (영업일).
    top_n : 상위 몇 개 자산.
    cash_when_no_momentum : True면 양의 모멘텀 자산이 하나도 없을 때 전액 현금.
    """
    def strategy(prices: pd.DataFrame) -> pd.Series:
        cols = list(prices.columns)
        if len(prices) <= lookback_days or len(cols) == 0:
            return pd.Series({"CASH": 1.0})
        ret = prices.iloc[-1] / prices.iloc[-(lookback_days + 1)] - 1
        ret = ret.dropna()
        if len(ret) == 0:
            return pd.Series({"CASH": 1.0})

        if cash_when_no_momentum:
            positive = ret[ret > 0]
            if len(positive) == 0:
                return pd.Series({"CASH": 1.0})
            ranked = positive.sort_values(ascending=False)
        else:
            ranked = ret.sort_values(ascending=False)

        winners = ranked.head(top_n).index
        if len(winners) == 0:
            return pd.Series({"CASH": 1.0})
        w = pd.Series(0.0, index=cols)
        w.loc[winners] = 1.0 / len(winners)
        return w
    return strategy
