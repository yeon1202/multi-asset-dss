"""
펀더멘털 점수(0~100) & 상위 N 랭킹 테스트.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.scoring.fundamental_score import (
    HIGHER_IS_BETTER,
    fundamental_score,
    percentile_score,
    top_n,
    winsorize,
)


# ---------- winsorize ----------

def test_winsorize_clips_out_of_bounds_to_nan():
    df = pd.DataFrame({"per": [5, 200, 10], "pbr": [1, 2, 100]}, index=["A", "B", "C"])
    bounds = {"per": {"min": 0, "max": 100}, "pbr": {"min": 0, "max": 50}}
    out = winsorize(df, bounds)
    assert out.loc["A", "per"] == 5
    assert math.isnan(out.loc["B", "per"])
    assert math.isnan(out.loc["C", "pbr"])


# ---------- percentile_score ----------

def test_percentile_score_higher_better_ranks_max_at_top():
    s = pd.Series([1, 5, 3, 4, 2])
    p = percentile_score(s, higher_is_better=True)
    # 가장 큰 값(5)이 가장 높은 점수
    assert p.iloc[1] == 100.0
    # 가장 작은 값(1)이 가장 낮은 점수
    assert p.iloc[0] == 20.0


def test_percentile_score_lower_better_reverses():
    s = pd.Series([1, 5, 3, 4, 2])
    p = percentile_score(s, higher_is_better=False)
    # 1이 최고, 5가 최저
    assert p.iloc[0] == 80.0  # 100 - 20
    assert p.iloc[1] == 0.0


def test_percentile_score_keeps_nan():
    s = pd.Series([1, np.nan, 3, np.nan, 2])
    p = percentile_score(s, higher_is_better=True)
    assert math.isnan(p.iloc[1])
    assert math.isnan(p.iloc[3])
    # NaN 빼고 나머지(1, 3, 2)에 대해 백분위
    assert p.iloc[0] < p.iloc[4] < p.iloc[2]


# ---------- fundamental_score ----------

@pytest.fixture
def sample_ratios() -> pd.DataFrame:
    """
    가공된 5개 종목.
    A: 저PER·고ROE 우량주
    B: 평범
    C: 고PER·저ROE
    D: 저PER 인데 부채 많음
    E: NaN 많음
    """
    return pd.DataFrame(
        {
            "per":  [5,  10, 30, 6,  np.nan],
            "pbr":  [1,  2,  5,  1.5, 3],
            "roe":  [0.2, 0.1, 0.03, 0.15, np.nan],
            "opm":  [0.15, 0.1, 0.03, 0.05, 0.08],
            "debt": [0.3, 0.7, 1.2, 2.5, 0.5],
        },
        index=pd.Index(["A", "B", "C", "D", "E"], name="code"),
    )


def test_fundamental_score_composite_is_0_to_100(sample_ratios):
    weights = {"per": 0.2, "pbr": 0.15, "roe": 0.3, "opm": 0.2, "debt": 0.15}
    out = fundamental_score(sample_ratios, weights)
    scores = out["composite_score"].dropna()
    assert (scores >= 0).all() and (scores <= 100).all()


def test_fundamental_score_a_beats_c(sample_ratios):
    """A(우량) > C(고PER·저ROE) 가 직관적으로 맞아야 함."""
    weights = {"per": 0.2, "pbr": 0.15, "roe": 0.3, "opm": 0.2, "debt": 0.15}
    out = fundamental_score(sample_ratios, weights)
    assert out.loc["A", "composite_score"] > out.loc["C", "composite_score"]


def test_fundamental_score_handles_nan_via_renormalization(sample_ratios):
    """E 는 per, roe 가 NaN — 나머지 항목 + 가중치 재정규화 로 점수 산출."""
    weights = {"per": 0.2, "pbr": 0.15, "roe": 0.3, "opm": 0.2, "debt": 0.15}
    out = fundamental_score(sample_ratios, weights)
    e = out.loc["E", "composite_score"]
    assert not math.isnan(e)
    assert 0 <= e <= 100


def test_fundamental_score_returns_raw_ratios_and_subscores(sample_ratios):
    weights = {"per": 0.2, "pbr": 0.15, "roe": 0.3, "opm": 0.2, "debt": 0.15}
    out = fundamental_score(sample_ratios, weights)
    expected = ["per", "pbr", "roe", "opm", "debt",
                "per_score", "pbr_score", "roe_score", "opm_score", "debt_score",
                "composite_score"]
    assert list(out.columns) == expected


def test_fundamental_score_applies_sanity_bounds():
    """이상치 컷오프 후 점수 계산."""
    ratios = pd.DataFrame(
        {"per": [5, 200, 10], "pbr": [1, 2, 3], "roe": [0.1, 0.1, 0.1],
         "opm": [0.1, 0.1, 0.1], "debt": [0.5, 0.5, 0.5]},
        index=pd.Index(["A", "B", "C"], name="code"),
    )
    weights = {"per": 1.0, "pbr": 0, "roe": 0, "opm": 0, "debt": 0}
    bounds = {"per": {"min": 0, "max": 100}}
    out = fundamental_score(ratios, weights, sanity_bounds=bounds)
    # B 는 PER 200으로 컷오프 → per_score NaN → 가중치 재정규화 불가 → composite NaN
    assert math.isnan(out.loc["B", "composite_score"])
    # A, C 는 정상
    assert not math.isnan(out.loc["A", "composite_score"])


# ---------- top_n ----------

def test_top_n_basic(sample_ratios):
    weights = {"per": 0.2, "pbr": 0.15, "roe": 0.3, "opm": 0.2, "debt": 0.15}
    scored = fundamental_score(sample_ratios, weights)
    leaders = top_n(scored, n=2)
    assert len(leaders) == 2
    # 첫번째가 두번째보다 점수 ≥
    assert leaders["composite_score"].iloc[0] >= leaders["composite_score"].iloc[1]


def test_top_n_skips_nan_composite():
    """composite 가 NaN 인 행은 ranking 에서 제외."""
    scored = pd.DataFrame(
        {"composite_score": [80, np.nan, 60, 70, np.nan]},
        index=["A", "B", "C", "D", "E"],
    )
    out = top_n(scored, n=10)
    assert set(out.index) == {"A", "C", "D"}


def test_top_n_missing_column_raises():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(KeyError):
        top_n(df)


# ---------- HIGHER_IS_BETTER spec ----------

def test_higher_is_better_directions_match_intuition():
    assert HIGHER_IS_BETTER["roe"] is True
    assert HIGHER_IS_BETTER["opm"] is True
    assert HIGHER_IS_BETTER["per"] is False
    assert HIGHER_IS_BETTER["pbr"] is False
    assert HIGHER_IS_BETTER["debt"] is False
