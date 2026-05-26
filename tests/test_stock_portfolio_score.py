"""
Phase 8 주식 포트폴리오 점수 테스트.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.scoring.stock_portfolio_score import (
    sector_regime_scores,
    stock_composite_score,
)


# ---------- sector_regime_scores ----------

def test_sector_regime_scores_applies_preference():
    """반도체(1.0) × risk-on(+0.5) → 75, 통신(0.3) × risk-on(+0.5) → 57.5"""
    sector_map = {"A": "반도체", "B": "통신"}
    prefs = {"반도체": 1.0, "통신": 0.3}
    out = sector_regime_scores(sector_map, prefs, default_preference=0.5, regime_score=0.5)
    assert out["A"] == pytest.approx(75.0)  # 50 + 50*1.0*0.5
    assert out["B"] == pytest.approx(57.5)  # 50 + 50*0.3*0.5


def test_sector_regime_scores_default_preference():
    """매핑에 없는 섹터는 default 사용."""
    sector_map = {"X": "신생산업"}
    prefs = {}
    out = sector_regime_scores(sector_map, prefs, default_preference=0.7, regime_score=1.0)
    assert out["X"] == pytest.approx(85.0)  # 50 + 50*0.7*1.0


def test_sector_regime_scores_negative_regime():
    """risk-off 환경에서 방어주(0.3)는 risk-on 자산(1.0)보다 낫게."""
    sector_map = {"A": "반도체", "B": "통신"}
    prefs = {"반도체": 1.0, "통신": 0.3}
    # regime = -0.5 (risk-off)
    out = sector_regime_scores(sector_map, prefs, 0.5, regime_score=-0.5)
    # A: 50 + 50*1.0*(-0.5) = 25
    # B: 50 + 50*0.3*(-0.5) = 42.5
    assert out["A"] == pytest.approx(25.0)
    assert out["B"] == pytest.approx(42.5)
    assert out["B"] > out["A"]


# ---------- stock_composite_score ----------

@pytest.fixture
def sample_data():
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    # A: 우상향 (technical 높음), B: 평탄, C: 우하향
    prices = pd.DataFrame({
        "A": np.linspace(100, 130, 100),
        "B": np.linspace(100, 100, 100),
        "C": np.linspace(100, 80, 100),
    }, index=idx)
    # 펀더멘털: A는 평범, B는 우수, C는 적자(nan)
    fund_scores = pd.Series({"A": 50.0, "B": 90.0, "C": float("nan")})
    sector_map = {"A": "반도체", "B": "통신", "C": "바이오"}
    prefs = {"반도체": 1.0, "통신": 0.3, "바이오": 1.0}
    return prices, fund_scores, sector_map, prefs


def test_stock_composite_returns_four_columns(sample_data):
    prices, fund, sector_map, prefs = sample_data
    out = stock_composite_score(
        prices=prices, fundamental_scores=fund, regime_score=0.5,
        sector_map=sector_map, sector_preferences=prefs, default_preference=0.5,
        weights={"technical": 0.33, "fundamental": 0.34, "regime": 0.33},
    )
    assert set(out.columns) == {"technical", "fundamental", "regime", "composite"}
    assert len(out) == 3


def test_stock_composite_range_0_100(sample_data):
    prices, fund, sector_map, prefs = sample_data
    out = stock_composite_score(
        prices=prices, fundamental_scores=fund, regime_score=0.5,
        sector_map=sector_map, sector_preferences=prefs, default_preference=0.5,
        weights={"technical": 0.33, "fundamental": 0.34, "regime": 0.33},
    )
    valid = out["composite"].dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_stock_composite_handles_nan_fundamental(sample_data):
    """C 는 fundamental NaN — tech + regime 두 신호로만 점수 계산 (가중치 재정규화)."""
    prices, fund, sector_map, prefs = sample_data
    out = stock_composite_score(
        prices=prices, fundamental_scores=fund, regime_score=0.5,
        sector_map=sector_map, sector_preferences=prefs, default_preference=0.5,
        weights={"technical": 0.3, "fundamental": 0.4, "regime": 0.3},
    )
    c_score = out.loc["C", "composite"]
    assert pd.notna(c_score)
    assert 0 <= c_score <= 100


def test_stock_composite_b_strong_fundamental_beats_c(sample_data):
    """B (펀더멘털 우수, 통신 섹터) 가 C (적자, 가격 하락) 보다 점수 높아야."""
    prices, fund, sector_map, prefs = sample_data
    out = stock_composite_score(
        prices=prices, fundamental_scores=fund, regime_score=0.5,
        sector_map=sector_map, sector_preferences=prefs, default_preference=0.5,
        weights={"technical": 0.3, "fundamental": 0.4, "regime": 0.3},
    )
    assert out.loc["B", "composite"] > out.loc["C", "composite"]


def test_stock_composite_weight_normalization(sample_data):
    """가중치 합이 1 이 아니어도 내부 정규화."""
    prices, fund, sector_map, prefs = sample_data
    out = stock_composite_score(
        prices=prices, fundamental_scores=fund, regime_score=0.5,
        sector_map=sector_map, sector_preferences=prefs, default_preference=0.5,
        weights={"technical": 3, "fundamental": 4, "regime": 3},  # 합 10
    )
    valid = out["composite"].dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_stock_composite_zero_weights_raises(sample_data):
    prices, fund, sector_map, prefs = sample_data
    with pytest.raises(ValueError, match="가중치"):
        stock_composite_score(
            prices=prices, fundamental_scores=fund, regime_score=0.5,
            sector_map=sector_map, sector_preferences=prefs, default_preference=0.5,
            weights={"technical": 0, "fundamental": 0, "regime": 0},
        )


def test_stock_composite_no_common_codes():
    prices = pd.DataFrame({"X": [100, 101]}, index=pd.date_range("2024-01-01", periods=2))
    fund = pd.Series({"Y": 50.0})
    with pytest.raises(ValueError, match="공통 종목"):
        stock_composite_score(
            prices=prices, fundamental_scores=fund, regime_score=0.5,
            sector_map={"X": "반도체", "Y": "반도체"},
            sector_preferences={"반도체": 1.0}, default_preference=0.5,
            weights={"technical": 1, "fundamental": 1, "regime": 1},
        )
