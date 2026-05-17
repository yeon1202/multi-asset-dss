"""
마코위츠 평균-분산 포트폴리오 최적화 — Phase 4.

PyPortfolioOpt 사용:
  - EfficientFrontier 로 max-Sharpe 또는 min-volatility 비중 산출
  - 통합 점수(0-100) 를 기대수익률 mu 벡터로 매핑
  - 공분산은 과거 일간 수익률에서 추정
  - 제약: 자산별 최대 비중, 음수 비중 금지, sum = 1

PROJECT_SPEC.md §9.1 과적합 회피:
  - 단일 시점 결과는 의심. 백테스팅 (Phase 5) 에서 안정성 검증 필요.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from pypfopt import EfficientFrontier, expected_returns, risk_models


@dataclass(frozen=True)
class OptimizerResult:
    """최적화 결과 묶음."""

    weights: pd.Series                # index = 자산코드, value = [0, 1]
    expected_return: float            # 연환산 기대 수익률
    volatility: float                 # 연환산 변동성
    sharpe: float                     # 샤프 비율 (rf 차감 후)
    mu: pd.Series                     # 입력 기대수익률
    cov: pd.DataFrame                 # 입력 공분산 (연환산)


def score_to_expected_return(
    composite_score: pd.Series,
    max_expected_return: float,
) -> pd.Series:
    """
    composite_score(0-100) 를 기대수익률로 선형 매핑.

    매핑:
      score=50  → 0%
      score=100 → +max_expected_return
      score=0   → -max_expected_return

    이 매핑은 보수적인 시작점. Phase 5 백테스팅으로 calibration 필요.
    """
    centered = (composite_score - 50.0) / 50.0  # [-1, +1]
    return (centered * max_expected_return).rename("mu")


def estimate_covariance(
    prices: pd.DataFrame,
    annualize_factor: int = 252,
) -> pd.DataFrame:
    """과거 일간 수익률 기반 공분산 (연환산)."""
    return risk_models.sample_cov(prices, frequency=annualize_factor)


def optimize_max_sharpe(
    prices: pd.DataFrame,
    composite_score: pd.Series,
    max_expected_return: float,
    max_weight_per_asset: float = 0.40,
    min_weight_per_asset: float = 0.00,
    risk_free_rate: float = 0.035,
) -> OptimizerResult:
    """
    최대 샤프 비율 비중 산출.

    Parameters
    ----------
    prices : DataFrame
        columns = 자산코드, 일봉 종가 시계열.
    composite_score : Series
        자산별 0-100 통합 점수.
    max_expected_return : float
        score=100 일 때 연환산 기대수익률 상한.
    max_weight_per_asset : float
        한 자산이 가질 수 있는 최대 비중.

    Returns
    -------
    OptimizerResult
    """
    if len(composite_score) < 2:
        raise ValueError("자산이 2개 이상 필요")
    # 점수와 가격의 컬럼이 정렬되어 있어야 함
    common = composite_score.index.intersection(prices.columns)
    if len(common) < 2:
        raise ValueError("점수·가격 공통 자산이 2개 미만")
    prices = prices[common].dropna(how="any")
    score = composite_score.loc[common]

    mu = score_to_expected_return(score, max_expected_return)
    cov = estimate_covariance(prices)

    # EfficientFrontier — weight_bounds 로 자산별 [min, max] 제약
    ef = EfficientFrontier(
        mu, cov,
        weight_bounds=(min_weight_per_asset, max_weight_per_asset),
    )
    ef.max_sharpe(risk_free_rate=risk_free_rate)
    cleaned = ef.clean_weights(cutoff=1e-4, rounding=4)
    ret, vol, sharpe = ef.portfolio_performance(
        verbose=False, risk_free_rate=risk_free_rate
    )

    weights = pd.Series(cleaned, name="weight").reindex(common).fillna(0.0)
    return OptimizerResult(
        weights=weights,
        expected_return=float(ret),
        volatility=float(vol),
        sharpe=float(sharpe),
        mu=mu, cov=cov,
    )


def equal_weight_fallback(
    composite_score: pd.Series,
) -> pd.Series:
    """최적화 실패 시 동일 가중 (조용히 fallback)."""
    n = len(composite_score)
    return pd.Series(1.0 / n, index=composite_score.index, name="weight")
