"""
공시 정성 점수 집계 — Phase 6.

여러 공시 분석을 종목별 0-100 점수로 변환.
sentiment_score (-1~+1) 를 50 ± 50 매핑 + 가중 평균 (최근 공시 가중치 ↑).
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Sequence

import numpy as np
import pandas as pd


def sentiment_to_score(sentiment: float) -> float:
    """sentiment (-1~+1) → 0-100 점수. 50 = 중립."""
    if sentiment is None or (isinstance(sentiment, float) and math.isnan(sentiment)):
        return 50.0
    s = max(-1.0, min(1.0, float(sentiment)))
    return 50.0 + 50.0 * s


def time_decay_weight(days_ago: int, half_life_days: int = 30) -> float:
    """
    최근 공시일수록 가중치 ↑. 지수 감쇠 (반감기 half_life_days).

    days_ago=0 → 1.0, days_ago=half_life → 0.5, days_ago=2*half_life → 0.25
    """
    if days_ago < 0:
        return 1.0
    return float(0.5 ** (days_ago / half_life_days))


def aggregate_disclosure_score(
    analyses: Sequence[dict],
    as_of: date | None = None,
    half_life_days: int = 30,
) -> float:
    """
    여러 공시 분석을 단일 점수(0-100)로 집계.

    Parameters
    ----------
    analyses : list of dict
        각각 {'sentiment_score': float, 'report_date': 'YYYY-MM-DD'} 필요.
    as_of : 기준일 (없으면 오늘).
    half_life_days : 시간 가중치 반감기.

    Returns
    -------
    float
        0-100 점수. 분석이 없으면 50 (중립).
    """
    if not analyses:
        return 50.0
    as_of = as_of or date.today()

    weighted_sum = 0.0
    total_weight = 0.0
    for a in analyses:
        sent = a.get("sentiment_score")
        report_date_str = a.get("report_date")
        if sent is None or report_date_str is None:
            continue
        try:
            rd = datetime.strptime(str(report_date_str), "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        days_ago = max(0, (as_of - rd).days)
        weight = time_decay_weight(days_ago, half_life_days)
        weighted_sum += sentiment_to_score(float(sent)) * weight
        total_weight += weight

    if total_weight == 0:
        return 50.0
    return weighted_sum / total_weight


def disclosure_score_table(
    analyses_by_code: dict[str, list[dict]],
    as_of: date | None = None,
    half_life_days: int = 30,
) -> pd.DataFrame:
    """
    종목별 집계 점수 테이블.

    Returns
    -------
    pd.DataFrame
        index = stock_code, columns = [n_disclosures, latest_score, time_weighted_score]
    """
    rows = []
    for code, analyses in analyses_by_code.items():
        if not analyses:
            rows.append({"code": code, "n_disclosures": 0,
                         "latest_score": 50.0, "time_weighted_score": 50.0})
            continue
        latest = max(
            analyses,
            key=lambda a: str(a.get("report_date", "")),
        )
        rows.append({
            "code": code,
            "n_disclosures": len(analyses),
            "latest_score": sentiment_to_score(float(latest.get("sentiment_score", 0.0))),
            "time_weighted_score": aggregate_disclosure_score(
                analyses, as_of, half_life_days
            ),
        })
    return pd.DataFrame(rows).set_index("code")
