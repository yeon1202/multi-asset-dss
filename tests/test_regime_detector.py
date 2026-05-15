"""
레짐 분류기 단위 테스트.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.regime.detector import (
    classify_regime,
    detect_history,
    feature_score,
)


# ---------- feature_score ----------

def test_feature_score_inverse_direction_low_is_risk_on():
    """direction=-1: 값이 lo 이하면 +1 (risk-on)."""
    # VIX 가 10이면 lo=15 미만 → risk-on
    assert feature_score(10, lo=15, hi=30, direction=-1) == pytest.approx(1.0)


def test_feature_score_inverse_direction_high_is_risk_off():
    """direction=-1: 값이 hi 이상이면 -1 (risk-off)."""
    assert feature_score(40, lo=15, hi=30, direction=-1) == pytest.approx(-1.0)


def test_feature_score_inverse_direction_midpoint_is_zero():
    """direction=-1: 중간값은 0."""
    assert feature_score(22.5, lo=15, hi=30, direction=-1) == pytest.approx(0.0)


def test_feature_score_positive_direction_low_is_risk_off():
    """direction=+1 (예: 스프레드 정상화): 값이 낮으면 risk-off."""
    assert feature_score(-1, lo=-0.5, hi=1.5, direction=1) == pytest.approx(-1.0)
    assert feature_score(2.0, lo=-0.5, hi=1.5, direction=1) == pytest.approx(1.0)


def test_feature_score_nan_input():
    assert math.isnan(feature_score(float("nan"), lo=0, hi=1, direction=-1))


def test_feature_score_invalid_direction():
    with pytest.raises(ValueError):
        feature_score(10, lo=0, hi=1, direction=0)


def test_feature_score_invalid_bounds():
    with pytest.raises(ValueError):
        feature_score(10, lo=5, hi=5, direction=-1)


# ---------- classify_regime ----------

@pytest.fixture
def sample_config() -> dict:
    return {
        "features": {
            "vix":     {"lo": 15, "hi": 30,  "direction": -1, "weight": 0.5},
            "hy_oas":  {"lo": 3,  "hi": 6,   "direction": -1, "weight": 0.3},
            "spread":  {"lo": -0.5, "hi": 1.5, "direction": 1, "weight": 0.2},
        },
        "thresholds": {"risk_on": 0.3, "risk_off": -0.3},
    }


def test_classify_regime_strong_risk_on(sample_config):
    """모든 feature 가 risk-on 방향."""
    values = {"vix": 12, "hy_oas": 2.0, "spread": 2.0}
    r = classify_regime(values, sample_config)
    assert r.score == pytest.approx(1.0)
    assert r.label == "risk_on"


def test_classify_regime_strong_risk_off(sample_config):
    values = {"vix": 35, "hy_oas": 7.0, "spread": -1.0}
    r = classify_regime(values, sample_config)
    assert r.score == pytest.approx(-1.0)
    assert r.label == "risk_off"


def test_classify_regime_neutral(sample_config):
    """중간 값 → 0 근처."""
    values = {"vix": 22.5, "hy_oas": 4.5, "spread": 0.5}
    r = classify_regime(values, sample_config)
    assert abs(r.score) < 0.01
    assert r.label == "neutral"


def test_classify_regime_nan_feature_ignored(sample_config):
    """NaN feature 는 가중치 재정규화로 제외."""
    values = {"vix": 12, "hy_oas": float("nan"), "spread": 2.0}
    r = classify_regime(values, sample_config)
    # vix(0.5/0.7) + spread(0.2/0.7) 가중 → 둘 다 +1 → 결과 +1
    assert r.score == pytest.approx(1.0)
    assert r.label == "risk_on"
    assert math.isnan(r.feature_scores["hy_oas"])
    assert "hy_oas" not in r.contributions


def test_classify_regime_all_nan_returns_nan_neutral(sample_config):
    values = {"vix": float("nan"), "hy_oas": float("nan"), "spread": float("nan")}
    r = classify_regime(values, sample_config)
    assert math.isnan(r.score)
    assert r.label == "neutral"


def test_classify_regime_contributions_sum_to_score(sample_config):
    """기여도 합 = 종합 점수 (가중치 정규화 후)."""
    values = {"vix": 18, "hy_oas": 5.0, "spread": 0.0}
    r = classify_regime(values, sample_config)
    assert sum(r.contributions.values()) == pytest.approx(r.score)


# ---------- detect_history ----------

def test_detect_history_basic(sample_config):
    df = pd.DataFrame(
        {
            "vix": [12, 22.5, 35],
            "hy_oas": [2.0, 4.5, 7.0],
            "spread": [2.0, 0.5, -1.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    hist = detect_history(df, sample_config)
    assert list(hist["label"]) == ["risk_on", "neutral", "risk_off"]
    assert hist["score"].iloc[0] > 0
    assert hist["score"].iloc[2] < 0


def test_detect_history_empty_input(sample_config):
    df = pd.DataFrame()
    hist = detect_history(df, sample_config)
    assert hist.empty


def test_detect_history_columns_have_subscores(sample_config):
    df = pd.DataFrame(
        {"vix": [20], "hy_oas": [4], "spread": [0.5]},
        index=pd.date_range("2024-01-01", periods=1, freq="D"),
    )
    hist = detect_history(df, sample_config)
    for col in ("score", "label", "vix_score", "hy_oas_score", "spread_score"):
        assert col in hist.columns
