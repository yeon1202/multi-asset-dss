"""
자산 레짐 적합도 점수 테스트.
"""
from __future__ import annotations

import math

import pytest

from src.scoring.regime_fit import asset_fit_score, asset_fit_table


def test_perfect_match_risk_on():
    """risk-on 자산(+1) × risk-on 레짐(+1) → 100."""
    assert asset_fit_score(1.0, 1.0) == pytest.approx(100.0)


def test_perfect_mismatch():
    """risk-on 자산 × risk-off 레짐 → 0."""
    assert asset_fit_score(1.0, -1.0) == pytest.approx(0.0)


def test_risk_off_asset_likes_risk_off_regime():
    """risk-off 자산(-1) × risk-off 레짐(-1) → 100."""
    assert asset_fit_score(-1.0, -1.0) == pytest.approx(100.0)


def test_neutral_asset_always_fifty():
    """선호도 0 인 자산은 항상 50."""
    assert asset_fit_score(0.0, 1.0) == pytest.approx(50.0)
    assert asset_fit_score(0.0, -1.0) == pytest.approx(50.0)
    assert asset_fit_score(0.0, 0.0) == pytest.approx(50.0)


def test_neutral_regime_always_fifty():
    """레짐 점수 0 이면 모든 자산이 50."""
    assert asset_fit_score(1.0, 0.0) == pytest.approx(50.0)
    assert asset_fit_score(-1.0, 0.0) == pytest.approx(50.0)


def test_partial_preference():
    """선호도 0.5, 레짐 +0.5 → 50 + 50*0.25 = 62.5."""
    assert asset_fit_score(0.5, 0.5) == pytest.approx(62.5)


def test_fit_score_nan_inputs():
    assert math.isnan(asset_fit_score(float("nan"), 1.0))
    assert math.isnan(asset_fit_score(1.0, float("nan")))


def test_fit_table_sorted_by_match():
    """risk-on 레짐에서 risk-on 자산이 상위."""
    prefs = {
        "STOCK_KR": 1.0,
        "STOCK_US": 0.8,
        "BOND":     -0.7,
        "GOLD":     -0.5,
        "CASH":     0.0,
    }
    out = asset_fit_table(prefs, regime_score=1.0)
    # 정렬: 인덱스 순 그대로
    assert out.loc["STOCK_KR", "regime_fit"] == pytest.approx(100.0)
    assert out.loc["BOND", "regime_fit"] == pytest.approx(15.0)
    assert out.loc["CASH", "regime_fit"] == pytest.approx(50.0)


def test_fit_table_risk_off_regime():
    prefs = {"STOCK": 1.0, "BOND": -1.0, "CASH": 0.0}
    out = asset_fit_table(prefs, regime_score=-1.0)
    assert out.loc["STOCK", "regime_fit"] == pytest.approx(0.0)
    assert out.loc["BOND", "regime_fit"] == pytest.approx(100.0)
    assert out.loc["CASH", "regime_fit"] == pytest.approx(50.0)
