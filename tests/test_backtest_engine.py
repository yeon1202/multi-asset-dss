"""
백테스트 엔진 통합 테스트.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import CostConfig
from src.backtest.engine import run_backtest
from src.backtest.strategies import buy_and_hold, equal_weight, momentum_top_n


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """3년치 가격: A는 우상향, B는 평탄, C는 우하향."""
    idx = pd.date_range("2020-01-01", "2022-12-31", freq="B")
    n = len(idx)
    return pd.DataFrame({
        "A": np.linspace(100, 200, n),
        "B": np.linspace(100, 100, n),
        "C": np.linspace(100, 70, n),
    }, index=idx)


def test_buy_and_hold_matches_asset_return(sample_prices):
    """A buy-and-hold 의 누적수익률 ≈ A 의 가격 상승률 (- 비용)."""
    result = run_backtest(
        prices=sample_prices,
        strategy_fn=buy_and_hold("A"),
        rebalance_freq="M",
        initial_capital=1_000_000,
        cost_config=CostConfig(commission_rate=0.0, slippage_rate=0.0, tax_rate_sell=0.0),
    )
    # A 는 100 → 200 → 100% 상승
    assert result.metrics.total_return == pytest.approx(1.0, abs=0.01)


def test_buy_and_hold_with_costs_lower_return(sample_prices):
    """비용이 있으면 누적 수익률이 약간 줄어야 함."""
    no_cost = run_backtest(
        prices=sample_prices, strategy_fn=buy_and_hold("A"),
        rebalance_freq="M", initial_capital=1_000_000,
        cost_config=CostConfig(commission_rate=0.0, slippage_rate=0.0, tax_rate_sell=0.0),
    )
    with_cost = run_backtest(
        prices=sample_prices, strategy_fn=buy_and_hold("A"),
        rebalance_freq="M", initial_capital=1_000_000,
        cost_config=CostConfig(commission_rate=0.001, slippage_rate=0.001, tax_rate_sell=0.0),
    )
    # buy-and-hold 도 첫 매수에 비용 발생 → 약간 낮은 수익률
    assert with_cost.metrics.total_return <= no_cost.metrics.total_return


def test_equal_weight_finishes_between_assets(sample_prices):
    """A(+100%), B(0%), C(-30%) → 동일가중은 약 +23% 부근."""
    result = run_backtest(
        prices=sample_prices, strategy_fn=equal_weight(),
        rebalance_freq="M", initial_capital=1_000_000,
        cost_config=CostConfig(commission_rate=0.0, slippage_rate=0.0, tax_rate_sell=0.0),
    )
    # 거래비용 없으니 단순 평균과 유사
    assert -0.05 < result.metrics.total_return < 0.40


def test_momentum_picks_winners(sample_prices):
    """모멘텀 전략이 A 에 비중 — buy-and-hold A 와 비슷한 결과."""
    result = run_backtest(
        prices=sample_prices,
        strategy_fn=momentum_top_n(lookback_days=63, top_n=1),
        rebalance_freq="M", initial_capital=1_000_000,
        cost_config=CostConfig(commission_rate=0.0, slippage_rate=0.0, tax_rate_sell=0.0),
    )
    # A 가 압도적으로 우상향 → 모멘텀 전략이 거의 항상 A 선택
    # buy-and-hold A 보다 약간 낮을 수 있지만 양수 수익률
    assert result.metrics.total_return > 0.5


def test_engine_returns_weights_history(sample_prices):
    result = run_backtest(
        prices=sample_prices, strategy_fn=equal_weight(),
        rebalance_freq="M", initial_capital=1_000_000,
    )
    # 월간 리밸런싱 — 36개월 부근
    assert len(result.weights_history) >= 30


def test_engine_empty_prices_raises():
    with pytest.raises(ValueError):
        run_backtest(prices=pd.DataFrame(), strategy_fn=equal_weight())


def test_engine_equity_monotonic_index(sample_prices):
    """equity 인덱스는 시간 순서대로 정렬되어야."""
    result = run_backtest(
        prices=sample_prices, strategy_fn=equal_weight(),
        rebalance_freq="M", initial_capital=1_000_000,
    )
    assert result.equity.index.is_monotonic_increasing


def test_engine_total_cost_tracked(sample_prices):
    """거래비용 합계가 양수로 기록."""
    result = run_backtest(
        prices=sample_prices,
        strategy_fn=momentum_top_n(lookback_days=63, top_n=2),
        rebalance_freq="M", initial_capital=1_000_000,
        cost_config=CostConfig(commission_rate=0.001, slippage_rate=0.001),
    )
    assert result.transaction_costs_total > 0
