"""
공시 점수 집계 테스트.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from src.scoring.disclosure_score import (
    aggregate_disclosure_score,
    disclosure_score_table,
    sentiment_to_score,
    time_decay_weight,
)


def test_sentiment_to_score_neutral_is_fifty():
    assert sentiment_to_score(0.0) == 50.0


def test_sentiment_to_score_max_is_hundred():
    assert sentiment_to_score(1.0) == 100.0


def test_sentiment_to_score_min_is_zero():
    assert sentiment_to_score(-1.0) == 0.0


def test_sentiment_to_score_clamps():
    assert sentiment_to_score(5.0) == 100.0
    assert sentiment_to_score(-99.0) == 0.0


def test_sentiment_to_score_nan_is_fifty():
    assert sentiment_to_score(float("nan")) == 50.0
    assert sentiment_to_score(None) == 50.0


def test_time_decay_weight_today_is_one():
    assert time_decay_weight(0) == 1.0


def test_time_decay_weight_half_life():
    """반감기 = half_life_days."""
    assert time_decay_weight(30, half_life_days=30) == pytest.approx(0.5)


def test_time_decay_weight_double_half_life():
    assert time_decay_weight(60, half_life_days=30) == pytest.approx(0.25)


def test_aggregate_empty_returns_neutral():
    assert aggregate_disclosure_score([]) == 50.0


def test_aggregate_single_positive():
    analyses = [{"sentiment_score": 1.0, "report_date": date.today().isoformat()}]
    score = aggregate_disclosure_score(analyses, as_of=date.today())
    assert score == pytest.approx(100.0)


def test_aggregate_recent_outweighs_old():
    """최근 +1, 오래된 -1 → 평균 위가 아닌, 최근에 가중."""
    today = date(2025, 6, 1)
    old = today - timedelta(days=180)
    analyses = [
        {"sentiment_score": -1.0, "report_date": old.isoformat()},
        {"sentiment_score": +1.0, "report_date": today.isoformat()},
    ]
    score = aggregate_disclosure_score(analyses, as_of=today, half_life_days=30)
    # 오래된 가중치 ≈ 0.5^6 = 0.0156, 최근 = 1.0
    # 가중 점수 ≈ (0*0.0156 + 100*1.0) / (0.0156+1.0) ≈ 98.5
    assert score > 90


def test_aggregate_ignores_invalid_dates():
    today = date.today()
    analyses = [
        {"sentiment_score": 0.5, "report_date": "INVALID-DATE"},
        {"sentiment_score": 0.5, "report_date": today.isoformat()},
    ]
    score = aggregate_disclosure_score(analyses, as_of=today)
    # 유효한 1건만 → 0.5 → 75
    assert score == pytest.approx(75.0)


def test_disclosure_score_table_no_analyses():
    out = disclosure_score_table({"005930": [], "000660": []})
    assert out.loc["005930", "n_disclosures"] == 0
    assert out.loc["005930", "latest_score"] == 50.0
    assert out.loc["005930", "time_weighted_score"] == 50.0


def test_disclosure_score_table_with_analyses():
    today = date(2025, 6, 1)
    analyses_by_code = {
        "005930": [
            {"sentiment_score": 0.8, "report_date": today.isoformat()},
            {"sentiment_score": -0.2, "report_date": (today - timedelta(days=60)).isoformat()},
        ],
        "000660": [
            {"sentiment_score": -0.5, "report_date": today.isoformat()},
        ],
    }
    out = disclosure_score_table(analyses_by_code, as_of=today, half_life_days=30)
    assert out.loc["005930", "n_disclosures"] == 2
    assert out.loc["005930", "latest_score"] == sentiment_to_score(0.8)
    assert out.loc["000660", "n_disclosures"] == 1
    assert out.loc["000660", "latest_score"] == sentiment_to_score(-0.5)
