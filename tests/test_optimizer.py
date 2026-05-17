"""
마코위츠 최적화 테스트.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio.optimizer import (
    equal_weight_fallback,
    estimate_covariance,
    optimize_max_sharpe,
    score_to_expected_return,
)


@pytest.fixture
def stable_prices() -> pd.DataFrame:
    """3개 자산 250일. 상이한 추세 + 약간의 노이즈."""
    rng = np.random.default_rng(7)
    n = 250
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    drift_a = 0.0008
    drift_b = 0.0003
    drift_c = -0.0001
    ret_a = drift_a + rng.standard_normal(n) * 0.010
    ret_b = drift_b + rng.standard_normal(n) * 0.008
    ret_c = drift_c + rng.standard_normal(n) * 0.012
    return pd.DataFrame({
        "A": 100 * np.exp(ret_a.cumsum()),
        "B": 100 * np.exp(ret_b.cumsum()),
        "C": 100 * np.exp(ret_c.cumsum()),
    }, index=idx)


def test_score_to_expected_return_mapping():
    s = pd.Series({"A": 100.0, "B": 50.0, "C": 0.0})
    mu = score_to_expected_return(s, max_expected_return=0.2)
    assert mu["A"] == pytest.approx(0.2)
    assert mu["B"] == pytest.approx(0.0)
    assert mu["C"] == pytest.approx(-0.2)


def test_estimate_covariance_symmetric_positive(stable_prices):
    cov = estimate_covariance(stable_prices)
    # 정방행렬, 대칭
    assert cov.shape == (3, 3)
    assert np.allclose(cov.values, cov.values.T)
    # 대각 성분(분산) > 0
    assert (np.diag(cov.values) > 0).all()


def test_optimize_max_sharpe_basic(stable_prices):
    scores = pd.Series({"A": 80.0, "B": 60.0, "C": 30.0})
    result = optimize_max_sharpe(
        prices=stable_prices,
        composite_score=scores,
        max_expected_return=0.2,
        max_weight_per_asset=0.6,
    )
    # 비중 합 ~ 1
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-3)
    # 모든 비중 [0, 0.6]
    assert (result.weights >= 0).all()
    assert (result.weights <= 0.60 + 1e-6).all()
    # 점수가 가장 높은 A 의 비중이 가장 큰 비중을 가져야 (현실적인 결과)
    assert result.weights["A"] >= result.weights["C"]


def test_optimize_max_sharpe_respects_max_weight(stable_prices):
    """max_weight_per_asset 가 엄격히 지켜져야."""
    # A 가 압도적이지만 모든 자산이 양의 점수여야 max_sharpe 가 feasible
    scores = pd.Series({"A": 100.0, "B": 70.0, "C": 60.0})
    result = optimize_max_sharpe(
        prices=stable_prices, composite_score=scores,
        max_expected_return=0.2, max_weight_per_asset=0.4,
        risk_free_rate=0.0,
    )
    assert result.weights["A"] <= 0.40 + 1e-6
    assert result.weights["B"] <= 0.40 + 1e-6
    assert result.weights["C"] <= 0.40 + 1e-6


def test_optimize_too_few_assets_raises(stable_prices):
    scores = pd.Series({"A": 50.0})  # 1개만
    with pytest.raises(ValueError):
        optimize_max_sharpe(
            prices=stable_prices, composite_score=scores,
            max_expected_return=0.2,
        )


def test_equal_weight_fallback_basic():
    scores = pd.Series({"A": 80.0, "B": 60.0, "C": 30.0})
    w = equal_weight_fallback(scores)
    assert np.allclose(w.values, 1 / 3)
    assert w.sum() == pytest.approx(1.0)
    assert list(w.index) == ["A", "B", "C"]
