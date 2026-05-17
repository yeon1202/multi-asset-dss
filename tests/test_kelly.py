"""
Half-Kelly 사이징 테스트.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from src.portfolio.kelly import (
    apply_kelly_sizing,
    half_kelly,
    kelly_fraction,
)


def test_kelly_fraction_positive_edge():
    """μ=0.10, σ=0.20, r_f=0 → f = 0.10 / 0.04 = 2.5."""
    assert kelly_fraction(0.10, 0.20, 0.0) == pytest.approx(2.5)


def test_kelly_fraction_subtract_rf():
    """μ=0.10, σ=0.20, r_f=0.04 → (0.10-0.04)/0.04 = 1.5."""
    assert kelly_fraction(0.10, 0.20, 0.04) == pytest.approx(1.5)


def test_kelly_fraction_negative_edge_zero():
    """기대수익 < 무위험 → 0 (공매도 안 함)."""
    assert kelly_fraction(0.02, 0.20, 0.05) == 0.0


def test_kelly_fraction_zero_vol_zero():
    assert kelly_fraction(0.10, 0.0, 0.0) == 0.0


def test_kelly_fraction_nan_return_zero():
    assert kelly_fraction(float("nan"), 0.20, 0.0) == 0.0


def test_half_kelly_is_half_with_default():
    full = kelly_fraction(0.10, 0.20, 0.0)
    h = half_kelly(0.10, 0.20, 0.0, fraction=0.5, cap=10.0)
    assert h == pytest.approx(full * 0.5)


def test_half_kelly_cap_enforced():
    """cap=0.95 면 Kelly 가 2.5*0.5=1.25 여도 0.95 로 잘림."""
    h = half_kelly(0.10, 0.20, 0.0, fraction=0.5, cap=0.95)
    assert h == 0.95


# ---------- apply_kelly_sizing ----------

def test_apply_kelly_sizing_high_sharpe_max_risk():
    """샤프 매우 높음 → Kelly 도 큼 → cap 까지 risk 가져감."""
    risky = pd.Series({"A": 0.5, "B": 0.5})
    out = apply_kelly_sizing(
        risky_weights=risky,
        expected_return=0.30,
        volatility=0.15,
        risk_free_rate=0.0,
        kelly_fraction_param=0.5,
        max_total_risk_weight=0.95,
        min_cash_weight=0.05,
    )
    # cap=0.95 적용 → A=0.475, B=0.475, CASH=0.05
    assert out["A"] == pytest.approx(0.475, abs=1e-3)
    assert out["B"] == pytest.approx(0.475, abs=1e-3)
    assert out["CASH"] == pytest.approx(0.05, abs=1e-3)


def test_apply_kelly_sizing_low_sharpe_more_cash():
    """기대수익 낮음 → Kelly 낮음 → 현금 비중 커짐."""
    risky = pd.Series({"A": 0.5, "B": 0.5})
    out = apply_kelly_sizing(
        risky_weights=risky,
        expected_return=0.02,
        volatility=0.20,
        risk_free_rate=0.0,
        kelly_fraction_param=0.5,
        max_total_risk_weight=0.95,
        min_cash_weight=0.05,
    )
    # f = 0.02/0.04 = 0.5, half = 0.25
    # A=0.125, B=0.125, CASH=0.75
    assert out["A"] == pytest.approx(0.125, abs=1e-3)
    assert out["CASH"] == pytest.approx(0.75, abs=1e-3)


def test_apply_kelly_sizing_sums_to_one():
    risky = pd.Series({"A": 0.3, "B": 0.4, "C": 0.3})
    out = apply_kelly_sizing(
        risky_weights=risky,
        expected_return=0.08, volatility=0.18,
        risk_free_rate=0.03, kelly_fraction_param=0.5,
        max_total_risk_weight=0.95, min_cash_weight=0.05,
    )
    assert out.sum() == pytest.approx(1.0, abs=1e-9)


def test_apply_kelly_sizing_min_cash_enforced():
    """기대수익이 매우 높아도 min_cash 가 항상 지켜짐."""
    risky = pd.Series({"A": 0.5, "B": 0.5})
    out = apply_kelly_sizing(
        risky_weights=risky,
        expected_return=1.0, volatility=0.10,  # 비현실적으로 높은 샤프
        risk_free_rate=0.0, kelly_fraction_param=0.5,
        max_total_risk_weight=0.99, min_cash_weight=0.10,
    )
    assert out["CASH"] >= 0.10 - 1e-6
