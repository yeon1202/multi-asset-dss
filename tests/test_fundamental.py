"""
펀더멘털 지표(PER/PBR/ROE/OPM/Debt) 단위 테스트.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.indicators.fundamental import (
    compute_ratios,
    compute_ratios_table,
    debt_ratio,
    opm,
    pbr,
    per,
    roe,
)


# ---------- per ----------

def test_per_normal():
    # 시총 1조, 순이익 1천억 → PER 10
    assert per(1_000_000_000_000, 100_000_000_000) == pytest.approx(10.0)


def test_per_negative_earnings_is_nan():
    """적자(순이익 ≤ 0)는 PER 정의 불가."""
    assert math.isnan(per(1_000_000_000_000, -50_000_000_000))
    assert math.isnan(per(1_000_000_000_000, 0))


def test_per_nan_propagates():
    assert math.isnan(per(math.nan, 1.0))
    assert math.isnan(per(1.0, math.nan))


# ---------- pbr ----------

def test_pbr_normal():
    assert pbr(2_000_000_000_000, 1_000_000_000_000) == pytest.approx(2.0)


def test_pbr_negative_equity_is_nan():
    """자본잠식 시 PBR 정의 불가."""
    assert math.isnan(pbr(1e12, -1e11))


# ---------- roe ----------

def test_roe_positive():
    assert roe(100_000_000_000, 1_000_000_000_000) == pytest.approx(0.10)


def test_roe_negative_earnings_kept():
    """적자는 음수 ROE 그대로."""
    assert roe(-100, 1000) == pytest.approx(-0.10)


def test_roe_negative_equity_is_nan():
    assert math.isnan(roe(100, -1000))


# ---------- opm ----------

def test_opm_normal():
    assert opm(200, 1000) == pytest.approx(0.20)


def test_opm_zero_revenue_is_nan():
    assert math.isnan(opm(100, 0))
    assert math.isnan(opm(100, -1))


# ---------- debt_ratio ----------

def test_debt_ratio_normal():
    # 부채 200, 자본 100 → 2.0 (200%)
    assert debt_ratio(200, 100) == pytest.approx(2.0)


def test_debt_ratio_zero_equity_is_nan():
    assert math.isnan(debt_ratio(100, 0))


# ---------- compute_ratios ----------

def test_compute_ratios_all_keys():
    out = compute_ratios(
        market_cap=1e12,
        revenue=1e12,
        op_profit=2e11,
        net_income=1e11,
        total_assets=2e12,
        total_debt=8e11,
        total_equity=1.2e12,
    )
    assert set(out.keys()) == {"per", "pbr", "roe", "opm", "debt"}
    assert out["per"] == pytest.approx(10.0)
    assert out["pbr"] == pytest.approx(1e12 / 1.2e12)
    assert out["roe"] == pytest.approx(1e11 / 1.2e12)
    assert out["opm"] == pytest.approx(0.20)
    assert out["debt"] == pytest.approx(8e11 / 1.2e12)


# ---------- compute_ratios_table ----------

def test_compute_ratios_table_joins_correctly():
    fin = pd.DataFrame(
        {
            "revenue":      [1000, 2000, 500],
            "op_profit":    [200,  100,  -50],
            "net_income":   [100,  50,   -100],
            "total_assets": [3000, 4000, 1000],
            "total_debt":   [1500, 3000, 800],
            "total_equity": [1500, 1000, 200],
        },
        index=pd.Index(["A", "B", "C"], name="code"),
    )
    mcap = pd.DataFrame(
        {"market_cap": [1000, 800, 100]},
        index=pd.Index(["A", "B", "C"], name="code"),
    )
    out = compute_ratios_table(fin, mcap)
    assert list(out.columns) == ["per", "pbr", "roe", "opm", "debt"]
    assert out.loc["A", "per"] == pytest.approx(10.0)
    assert out.loc["A", "opm"] == pytest.approx(0.2)
    # C는 적자 → PER NaN, ROE 음수
    assert math.isnan(out.loc["C", "per"])
    assert out.loc["C", "roe"] == pytest.approx(-0.5)


def test_compute_ratios_table_missing_market_cap():
    """시총 누락(NaN)이면 PER·PBR 만 NaN, 나머지는 정상 계산."""
    fin = pd.DataFrame(
        {
            "revenue":      [1000],
            "op_profit":    [100],
            "net_income":   [50],
            "total_assets": [2000],
            "total_debt":   [800],
            "total_equity": [1200],
        },
        index=pd.Index(["X"], name="code"),
    )
    mcap = pd.DataFrame({"market_cap": [np.nan]}, index=pd.Index(["X"], name="code"))
    out = compute_ratios_table(fin, mcap)
    assert math.isnan(out.loc["X", "per"])
    assert math.isnan(out.loc["X", "pbr"])
    assert out.loc["X", "roe"] == pytest.approx(50 / 1200)
    assert out.loc["X", "opm"] == pytest.approx(0.1)
