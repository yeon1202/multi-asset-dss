"""
거래 비용 모델 테스트.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.costs import CostConfig, trade_cost, turnover


def test_turnover_no_change():
    w = pd.Series({"A": 0.5, "B": 0.5})
    assert turnover(w, w) == pytest.approx(0.0)


def test_turnover_full_swap():
    """A→B 완전 교체."""
    w1 = pd.Series({"A": 1.0, "B": 0.0})
    w2 = pd.Series({"A": 0.0, "B": 1.0})
    assert turnover(w1, w2) == pytest.approx(1.0)


def test_turnover_partial():
    w1 = pd.Series({"A": 0.5, "B": 0.5})
    w2 = pd.Series({"A": 0.7, "B": 0.3})
    assert turnover(w1, w2) == pytest.approx(0.2)


def test_turnover_different_indices():
    """한쪽에만 있는 자산도 처리."""
    w1 = pd.Series({"A": 1.0})
    w2 = pd.Series({"B": 1.0})
    assert turnover(w1, w2) == pytest.approx(1.0)


def test_trade_cost_no_change_zero():
    w = pd.Series({"A": 1.0})
    cfg = CostConfig()
    assert trade_cost(w, w, 1_000_000, cfg) == pytest.approx(0.0)


def test_trade_cost_full_swap():
    """비중 1.0 매도 + 비중 1.0 매수 → 거래량 2.0."""
    w1 = pd.Series({"A": 1.0, "B": 0.0})
    w2 = pd.Series({"A": 0.0, "B": 1.0})
    cfg = CostConfig(commission_rate=0.001, slippage_rate=0.001, tax_rate_sell=0.0)
    cost = trade_cost(w1, w2, 1_000_000, cfg)
    # (0.001 + 0.001) * 2.0 * 1_000_000 = 4000
    assert cost == pytest.approx(4000.0)


def test_trade_cost_with_sell_tax():
    """매도분에만 거래세 부과."""
    w1 = pd.Series({"A": 1.0, "B": 0.0})
    w2 = pd.Series({"A": 0.5, "B": 0.5})
    cfg = CostConfig(commission_rate=0.0, slippage_rate=0.0, tax_rate_sell=0.01)
    cost = trade_cost(w1, w2, 1_000_000, cfg)
    # 매도 0.5만, tax 1% → 5000
    assert cost == pytest.approx(5000.0)


def test_trade_cost_zero_value_zero_cost():
    w1 = pd.Series({"A": 0.0})
    w2 = pd.Series({"A": 1.0})
    cfg = CostConfig()
    assert trade_cost(w1, w2, 0.0, cfg) == 0.0
