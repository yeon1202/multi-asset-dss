"""
다자산 통합 점수 테스트.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.scoring.composite_score import (
    composite_score,
    momentum_returns,
    regime_fit_score,
    technical_score,
    vol_adjusted_momentum,
)


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """3개 자산, 100일. A는 상승, B는 평탄, C는 하락."""
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    return pd.DataFrame({
        "A": np.linspace(100, 130, 100),  # +30%
        "B": np.linspace(100, 100, 100),  # 0%
        "C": np.linspace(100, 90, 100),   # -10%
    }, index=idx)


def test_momentum_returns_basic(sample_prices):
    r = momentum_returns(sample_prices, lookback_days=99)
    assert r["A"] > 0.25
    assert abs(r["B"]) < 1e-9
    assert r["C"] < -0.05


def test_momentum_insufficient_data():
    """데이터가 lookback 이하이면 0."""
    df = pd.DataFrame({"A": [100, 101]}, index=pd.date_range("2024-01-01", periods=2))
    r = momentum_returns(df, lookback_days=10)
    assert r["A"] == 0.0


def test_technical_score_ranks_assets(sample_prices):
    s = technical_score(sample_prices, lookback_days=99, use_vol_adjusted=False)
    # A 가 최고, C 가 최저
    assert s["A"] > s["B"]
    assert s["B"] > s["C"]
    assert 0 <= s.min() and s.max() <= 100


def test_technical_score_single_asset_returns_fifty():
    df = pd.DataFrame({"A": np.linspace(100, 110, 50)})
    s = technical_score(df, lookback_days=20)
    assert s["A"] == 50.0


def test_vol_adjusted_momentum_high_vol_reduces_score():
    """변동성이 높은 자산은 같은 수익률이라도 점수가 낮아야."""
    idx = pd.date_range("2024-01-01", periods=200)
    # A: 안정적으로 +20%
    a = np.linspace(100, 120, 200)
    # B: 같은 +20% 인데 변동성 큼
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(200) * 5
    b = np.linspace(100, 120, 200) + noise
    df = pd.DataFrame({"A": a, "B": b}, index=idx)
    adj = vol_adjusted_momentum(df, lookback_days=63)
    # A 의 vol-adjusted 가 B 보다 높아야 (같은 수익률, 더 낮은 변동성)
    assert adj["A"] > adj["B"]


def test_regime_fit_score_matches_phase3():
    """Phase 3 적합도 식과 동일해야 (재구현 회귀)."""
    prefs = {"STOCK": 1.0, "BOND": -0.7, "CASH": 0.0}
    out = regime_fit_score(regime_score=0.5, preferences=prefs)
    assert out["STOCK"] == pytest.approx(75.0)   # 50 + 50*1*0.5
    assert out["BOND"] == pytest.approx(32.5)    # 50 + 50*(-0.7)*0.5
    assert out["CASH"] == pytest.approx(50.0)


def test_composite_score_columns(sample_prices):
    prefs = {"A": 1.0, "B": 0.0, "C": -1.0}
    out = composite_score(
        prices=sample_prices,
        regime_score=0.5,
        preferences=prefs,
        weights={"technical": 0.5, "regime": 0.5},
        technical_cfg={"momentum_lookback_days": 60, "use_vol_adjusted": False},
    )
    assert list(out.columns) == ["technical", "regime", "composite"]
    assert (out["composite"] >= 0).all() and (out["composite"] <= 100).all()


def test_composite_score_weight_normalization(sample_prices):
    """가중치 합이 1 이 아니어도 내부 정규화 → composite 가 0-100 유지."""
    prefs = {"A": 1.0, "B": 0.0, "C": -1.0}
    out = composite_score(
        prices=sample_prices,
        regime_score=0.5,
        preferences=prefs,
        weights={"technical": 2.0, "regime": 8.0},  # 비정규
    )
    assert (out["composite"] >= 0).all() and (out["composite"] <= 100).all()


def test_composite_score_zero_weights_raises(sample_prices):
    prefs = {"A": 1.0, "B": 0.0, "C": -1.0}
    with pytest.raises(ValueError):
        composite_score(
            prices=sample_prices,
            regime_score=0.5,
            preferences=prefs,
            weights={"technical": 0, "regime": 0},
        )


def test_composite_score_no_common_assets():
    prefs = {"X": 1.0, "Y": -1.0}
    prices = pd.DataFrame({"A": [100, 101], "B": [200, 201]},
                          index=pd.date_range("2024-01-01", periods=2))
    with pytest.raises(ValueError, match="공통 자산"):
        composite_score(
            prices=prices,
            regime_score=0.5,
            preferences=prefs,
            weights={"technical": 0.5, "regime": 0.5},
        )
