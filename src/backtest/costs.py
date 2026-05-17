"""
거래 비용 모델 — Phase 5.

PROJECT_SPEC.md §6 Phase 5: 수수료 0.015% + 슬리피지 0.05% + 세금 반영.
Phase 1 유니버스는 ETF 라 거래세 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostConfig:
    """거래 비용 비율 — 양방향 적용 (매수·매도 둘 다)."""

    commission_rate: float = 0.00015   # 0.015%
    slippage_rate:   float = 0.0005    # 0.05%
    tax_rate_sell:   float = 0.0       # 매도 시 거래세 (ETF=0, 주식=0.0018)


def turnover(prev_weights: pd.Series, new_weights: pd.Series) -> float:
    """
    회전율 = Σ |Δw| / 2.

    두 비중의 인덱스가 다르면 outer-join 후 NaN→0.
    회전율 0 = 변화 없음, 1 = 완전 교체.
    """
    all_keys = prev_weights.index.union(new_weights.index)
    prev_aligned = prev_weights.reindex(all_keys, fill_value=0.0)
    new_aligned = new_weights.reindex(all_keys, fill_value=0.0)
    return float((prev_aligned - new_aligned).abs().sum() / 2.0)


def trade_cost(
    prev_weights: pd.Series,
    new_weights: pd.Series,
    portfolio_value: float,
    config: CostConfig,
) -> float:
    """
    리밸런싱에 들어가는 총 비용 (원).

    매매 회전율 × 포트폴리오 가치 × (수수료 + 슬리피지)
    + 매도 분에 대한 거래세 (있을 경우).

    이 비용은 포트폴리오 가치에서 즉시 차감.
    """
    if portfolio_value <= 0:
        return 0.0
    all_keys = prev_weights.index.union(new_weights.index)
    prev_aligned = prev_weights.reindex(all_keys, fill_value=0.0)
    new_aligned = new_weights.reindex(all_keys, fill_value=0.0)
    delta = new_aligned - prev_aligned

    # 양방향 비용 (수수료·슬리피지)
    two_way = (config.commission_rate + config.slippage_rate)
    buy_volume = delta.clip(lower=0).sum()        # 새로 산 비중
    sell_volume = (-delta.clip(upper=0)).sum()    # 새로 판 비중

    cost_pct = two_way * (buy_volume + sell_volume) + config.tax_rate_sell * sell_volume
    return float(cost_pct * portfolio_value)
