"""
기술적 지표 단위 테스트.

목적: 함수가 수학적으로 맞는지, 엣지 케이스(짧은 시계열, 평탄 시계열 등)에서
NaN을 적절히 반환하는지 검증.

pytest 명령: `pytest tests/test_technical.py -v`
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.indicators.technical import (
    daily_returns,
    rsi,
    sma,
    summarize,
    volatility,
)


@pytest.fixture
def linear_up() -> pd.Series:
    """100, 101, 102, ... 100일치 — 단조 증가."""
    return pd.Series(np.arange(100, 200, dtype=float), name="close")


@pytest.fixture
def linear_down() -> pd.Series:
    """200, 199, ... 단조 감소."""
    return pd.Series(np.arange(200, 100, -1, dtype=float), name="close")


@pytest.fixture
def flat() -> pd.Series:
    """모든 값이 100인 평탄 시계열."""
    return pd.Series([100.0] * 50, name="close")


# ---------- RSI ----------

def test_rsi_monotonic_up_approaches_100(linear_up):
    """계속 오르기만 하면 RSI는 100에 수렴."""
    r = rsi(linear_up, period=14)
    last = r.iloc[-1]
    assert not math.isnan(last)
    assert last == pytest.approx(100.0, abs=1e-6)


def test_rsi_monotonic_down_approaches_0(linear_down):
    """계속 내리기만 하면 RSI는 0에 수렴."""
    r = rsi(linear_down, period=14)
    last = r.iloc[-1]
    assert not math.isnan(last)
    assert last == pytest.approx(0.0, abs=1e-6)


def test_rsi_flat_series_is_nan_or_100(flat):
    """변동이 없으면 RSI 정의가 모호. 우리 구현은 손실=0이면 100."""
    r = rsi(flat, period=14)
    last = r.iloc[-1]
    # 손실도 이익도 0 → 우리 구현에서는 100으로 정의
    assert last == 100.0


def test_rsi_too_short_returns_all_nan():
    """기간보다 짧으면 전부 NaN."""
    short = pd.Series([1.0, 2.0, 3.0])
    r = rsi(short, period=14)
    assert r.isna().all()
    assert len(r) == len(short)


def test_rsi_invalid_period_raises():
    s = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        rsi(s, period=0)


def test_rsi_range_bounds():
    """랜덤 시계열에서도 RSI는 [0, 100] 범위."""
    rng = np.random.default_rng(42)
    s = pd.Series(100 + rng.standard_normal(200).cumsum())
    r = rsi(s, period=14).dropna()
    assert (r >= 0).all() and (r <= 100).all()


# ---------- SMA ----------

def test_sma_basic():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(s, window=3)
    # 첫 두 개는 NaN, 그 뒤는 [2, 3, 4]
    assert out.iloc[:2].isna().all()
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[3] == pytest.approx(3.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_sma_name():
    s = pd.Series([1.0, 2.0, 3.0])
    assert sma(s, window=2).name == "SMA_2"


def test_sma_invalid_window():
    with pytest.raises(ValueError):
        sma(pd.Series([1.0]), window=0)


# ---------- Returns & Volatility ----------

def test_daily_returns_first_is_nan():
    s = pd.Series([100.0, 110.0, 121.0])
    r = daily_returns(s)
    assert math.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(0.10)
    assert r.iloc[2] == pytest.approx(0.10)


def test_volatility_flat_is_zero(flat):
    """변동이 없으면 변동성도 0."""
    v = volatility(flat, window=20).dropna()
    assert (v == 0).all()


def test_volatility_annualization_factor():
    """factor를 바꾸면 √(factor) 배 차이."""
    rng = np.random.default_rng(0)
    s = pd.Series(100 + rng.standard_normal(300).cumsum())
    v252 = volatility(s, window=20, annualize_factor=252).iloc[-1]
    v365 = volatility(s, window=20, annualize_factor=365).iloc[-1]
    assert v365 / v252 == pytest.approx(math.sqrt(365 / 252), rel=1e-9)


def test_volatility_invalid_window():
    with pytest.raises(ValueError):
        volatility(pd.Series([1.0, 2.0]), window=1)


# ---------- summarize ----------

def test_summarize_keys():
    s = pd.Series(np.arange(100, 200, dtype=float))
    out = summarize(s, rsi_period=14, ma_short=20, ma_long=60)
    assert set(out.keys()) == {
        "last_close",
        "rsi",
        "sma_20",
        "sma_60",
        "vol_annualized",
    }
    assert out["last_close"] == 199.0
    assert out["rsi"] == pytest.approx(100.0, abs=1e-6)
