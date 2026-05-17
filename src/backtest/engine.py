"""
백테스트 엔진 — Phase 5.

기능:
  - 일별 가격 시계열을 받아 리밸런싱 시점마다 strategy_fn(prices_up_to_t) → weights 호출
  - 매 리밸런싱마다 turnover 기반 거래비용을 즉시 차감
  - 일별 포트폴리오 가치 곡선 + 성과 지표 산출

PROJECT_SPEC.md §7.2 look-ahead bias 금지 — strategy 에는 t 시점까지의 가격만 전달.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.backtest.costs import CostConfig, trade_cost
from src.backtest.metrics import PerformanceMetrics, compute_all


Strategy = Callable[[pd.DataFrame], pd.Series]


@dataclass
class BacktestResult:
    equity: pd.Series             # index = Date, 포트폴리오 가치
    weights_history: pd.DataFrame  # index = 리밸런싱일, columns = 자산
    metrics: PerformanceMetrics
    transaction_costs_total: float


def _rebalance_dates(prices: pd.DataFrame, freq: str) -> pd.DatetimeIndex:
    """리밸런싱 일자 — 영업일 기준 freq 의 마지막 날. 첫 영업일에도 항상 진입."""
    resample_rule = {"M": "ME", "Q": "QE", "W": "W-FRI", "D": "D"}.get(freq, freq)
    if freq == "D":
        return prices.index
    last_of_period = prices.resample(resample_rule).last().dropna(how="all").index
    aligned: list[pd.Timestamp] = []
    for ts in last_of_period:
        valid = prices.index[prices.index <= ts]
        if len(valid) > 0:
            aligned.append(valid[-1])
    # 첫 영업일에 강제 진입 — 그렇지 않으면 첫 리밸런싱 전까지 현금만 들고 있어 성과 왜곡
    if len(prices.index) > 0 and prices.index[0] not in aligned:
        aligned.insert(0, prices.index[0])
    return pd.DatetimeIndex(sorted(set(aligned)))


def run_backtest(
    prices: pd.DataFrame,
    strategy_fn: Strategy,
    rebalance_freq: str = "M",
    initial_capital: float = 10_000_000.0,
    cost_config: CostConfig | None = None,
    risk_free_rate: float = 0.0,
) -> BacktestResult:
    """
    Parameters
    ----------
    prices : DataFrame
        index = DatetimeIndex (일별 영업일), columns = 자산코드, 종가.
    strategy_fn : Strategy
        (prices_up_to_t) → weights Series.
    rebalance_freq : "M" | "Q" | "W" | "D"
    initial_capital : 초기 자본 (단위: 원)
    cost_config : CostConfig (None 이면 기본)
    risk_free_rate : 샤프 계산용 무위험 수익률

    Returns
    -------
    BacktestResult
    """
    if prices.empty:
        raise ValueError("prices 가 비어 있음")
    cost_cfg = cost_config or CostConfig()
    prices = prices.sort_index().ffill().dropna(how="all")

    rebalance_days = _rebalance_dates(prices, rebalance_freq)
    # 보유 비중 (현재 시점 — 가치 변동에 따라 시간이 지나며 drift 됨)
    holdings_value = pd.Series(0.0, index=list(prices.columns) + ["CASH"])
    holdings_value["CASH"] = initial_capital
    equity_history: list[tuple[pd.Timestamp, float]] = []
    weights_records: list[tuple[pd.Timestamp, pd.Series]] = []
    total_cost = 0.0

    prev_weights = pd.Series({"CASH": 1.0})

    for date in prices.index:
        today_prices = prices.loc[date]
        # 1) drift: 위험자산 가치는 가격 변동에 따라 갱신, 현금은 그대로
        if len(equity_history) > 0:
            prev_prices = prices.loc[equity_history[-1][0]]
            asset_codes = [c for c in holdings_value.index if c != "CASH"]
            ret = today_prices / prev_prices  # 비율
            for code in asset_codes:
                if code in ret.index and not pd.isna(ret[code]):
                    holdings_value[code] *= float(ret[code])

        portfolio_value = float(holdings_value.sum())

        # 2) 리밸런싱 일자면 weights 재산출
        if date in rebalance_days and portfolio_value > 0:
            history = prices.loc[:date]
            new_weights = strategy_fn(history)
            new_weights = new_weights.fillna(0.0)
            # 합이 1 이하 — 부족분은 현금
            if "CASH" not in new_weights.index:
                new_weights["CASH"] = max(0.0, 1.0 - new_weights.sum())

            cost = trade_cost(prev_weights, new_weights, portfolio_value, cost_cfg)
            total_cost += cost
            portfolio_value -= cost

            # 비중을 실제 가치로 환산
            for code in holdings_value.index:
                holdings_value[code] = portfolio_value * float(new_weights.get(code, 0.0))
            prev_weights = new_weights
            weights_records.append((date, new_weights.copy()))

        equity_history.append((date, float(holdings_value.sum())))

    equity = pd.Series({d: v for d, v in equity_history}).sort_index()
    weights_df = (
        pd.DataFrame({d: w for d, w in weights_records}).T
        if weights_records else pd.DataFrame()
    )
    metrics = compute_all(equity, risk_free_rate=risk_free_rate)
    return BacktestResult(
        equity=equity,
        weights_history=weights_df,
        metrics=metrics,
        transaction_costs_total=total_cost,
    )
