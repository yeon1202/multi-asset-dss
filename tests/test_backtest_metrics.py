"""
성과 지표 테스트.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import (
    annualized_volatility,
    cagr,
    calmar_ratio,
    compute_all,
    max_drawdown,
    monthly_win_rate,
    sharpe_ratio,
    sortino_ratio,
    total_return,
)


def _const_growth(start: float, daily_rate: float, days: int) -> pd.Series:
    """일정 일간 수익률로 자라는 시계열."""
    idx = pd.date_range("2020-01-01", periods=days, freq="B")
    values = start * (1 + daily_rate) ** np.arange(days)
    return pd.Series(values, index=idx)


def test_total_return_basic():
    s = pd.Series([100, 110, 120, 130])
    assert total_return(s) == pytest.approx(0.30)


def test_total_return_short():
    assert total_return(pd.Series([100])) == 0.0


def test_cagr_one_year():
    """1년간 20% 성장."""
    idx = pd.date_range("2020-01-01", "2020-12-31", freq="B")
    s = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    c = cagr(s)
    # 365일 기준 CAGR ≈ +20%
    assert 0.18 < c < 0.22


def test_annualized_vol_zero_for_flat():
    s = pd.Series([100.0] * 100, index=pd.date_range("2024-01-01", periods=100, freq="B"))
    assert annualized_volatility(s) == 0.0


def test_max_drawdown_signs():
    """100 → 120 → 80 → 100 → MDD = (80/120) - 1 = -0.333."""
    s = pd.Series([100, 120, 80, 100],
                  index=pd.date_range("2024-01-01", periods=4, freq="B"))
    mdd = max_drawdown(s)
    assert mdd == pytest.approx(-1 / 3, abs=1e-6)


def test_max_drawdown_monotonic_up_is_zero():
    s = _const_growth(100, 0.001, 100)
    assert max_drawdown(s) == pytest.approx(0.0)


def test_sharpe_ratio_zero_vol_zero():
    s = pd.Series([100.0] * 50, index=pd.date_range("2024-01-01", periods=50, freq="B"))
    assert sharpe_ratio(s) == 0.0


def test_sortino_returns_inf_for_no_downside():
    """하방 수익률 없으면 inf 또는 매우 큼."""
    s = _const_growth(100, 0.001, 100)
    val = sortino_ratio(s)
    assert math.isinf(val) or val > 100


def test_calmar_ratio_positive_growth_positive_mdd():
    """100 → 120 → 110 → 140."""
    s = pd.Series([100, 120, 110, 140],
                  index=pd.date_range("2024-01-01", periods=4, freq="B"))
    assert calmar_ratio(s) > 0


def test_monthly_win_rate_all_winning():
    """매월 +1% → 승률 100%."""
    idx = pd.date_range("2020-01-01", "2022-12-31", freq="B")
    s = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    wr = monthly_win_rate(s)
    assert wr == pytest.approx(1.0, abs=0.05)


def test_compute_all_returns_all_fields():
    s = _const_growth(100, 0.001, 250)
    m = compute_all(s, risk_free_rate=0.02)
    assert m.total_return > 0
    assert m.cagr > 0
    assert m.volatility >= 0
    assert m.max_drawdown <= 0
    assert m.n_obs == 250
