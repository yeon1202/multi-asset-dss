"""
해설(narrative) 생성기 테스트.
규칙 기반이라 결정론적 — 입력 → 출력 정확히 일치 가능.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.report.narrative import (
    FEATURE_LABELS,
    narrate_allocation,
    narrate_all,
    narrate_regime,
    narrate_scores,
    narrate_warnings,
)


# ---------- narrate_regime ----------

def test_regime_strong_risk_on():
    text = narrate_regime(
        label="risk_on",
        score=0.75,
        contributions={"vix": 0.30, "hy_oas": 0.25},
        feature_values={"vix": 12.0, "hy_oas": 2.5},
    )
    assert "강한" in text
    assert "위험선호" in text
    # 점수 표시
    assert "+0.75" in text
    # 가장 큰 기여 (vix)
    assert "VIX" in text or "vix" in text


def test_regime_weak_risk_off():
    text = narrate_regime(
        label="risk_off",
        score=-0.20,
        contributions={"vix": -0.15, "hy_oas": -0.05},
        feature_values={"vix": 28.0, "hy_oas": 4.5},
    )
    assert "약한" in text
    assert "위험회피" in text


def test_regime_neutral_close_to_zero():
    text = narrate_regime(
        label="neutral",
        score=0.05,
        contributions={"vix": 0.03, "hy_oas": 0.02},
        feature_values={"vix": 20.0, "hy_oas": 3.5},
    )
    assert "중립" in text


def test_regime_mentions_counter_signal():
    """강한 risk-on 안에 risk-off 신호가 있으면 단서 문구."""
    text = narrate_regime(
        label="risk_on",
        score=0.4,
        contributions={"vix": 0.30, "hy_oas": 0.25, "usd_krw": -0.15},
        feature_values={"vix": 14.0, "hy_oas": 2.5, "usd_krw": 1500.0},
    )
    assert "단" in text or "반대" in text


def test_regime_no_counter_signal_skipped():
    """모든 기여가 같은 방향이면 단서 문구 없음."""
    text = narrate_regime(
        label="risk_on",
        score=0.6,
        contributions={"vix": 0.3, "hy_oas": 0.3},
        feature_values={"vix": 12.0, "hy_oas": 2.5},
    )
    assert "단," not in text


# ---------- narrate_scores ----------

@pytest.fixture
def sample_scores():
    return pd.DataFrame({
        "technical": [85.0, 70.0, 50.0, 30.0, 15.0],
        "regime":    [70.0, 65.0, 55.0, 40.0, 35.0],
        "composite": [77.5, 67.5, 52.5, 35.0, 25.0],
    }, index=["A", "B", "C", "D", "E"])


def test_scores_picks_top_and_bottom(sample_scores):
    names = {"A": "에이", "B": "비", "C": "씨", "D": "디", "E": "이"}
    text = narrate_scores(sample_scores, names)
    assert "에이" in text
    assert "이" in text
    assert "77.5" in text  # 최고 점수
    assert "25.0" in text  # 최저 점수


def test_scores_high_momentum_mentioned(sample_scores):
    """기술적 점수 75 이상이면 모멘텀 언급."""
    text = narrate_scores(sample_scores, {"A": "에이"})
    assert "모멘텀" in text


def test_scores_distribution_summary(sample_scores):
    text = narrate_scores(sample_scores, {})
    # 60 이상이 2개, 40 미만이 2개
    assert "60점 이상" in text
    assert "40점 미만" in text


def test_scores_empty_returns_placeholder():
    text = narrate_scores(pd.DataFrame(), {})
    assert "없음" in text


# ---------- narrate_allocation ----------

def test_allocation_top_two_named():
    alloc = pd.Series({"A": 0.40, "B": 0.30, "C": 0.20, "CASH": 0.10})
    names = {"A": "에이", "B": "비", "C": "씨"}
    text = narrate_allocation(
        alloc, names,
        expected_return=0.10, volatility=0.15, sharpe=0.60, kelly_scale=0.80,
    )
    assert "에이" in text
    assert "비" in text
    assert "70.0%" in text  # 두 자산 합


def test_allocation_zero_weights_listed():
    alloc = pd.Series({"A": 0.95, "B": 0.0, "C": 0.0, "CASH": 0.05})
    names = {"A": "에이", "B": "비", "C": "씨"}
    text = narrate_allocation(
        alloc, names,
        expected_return=0.10, volatility=0.15, sharpe=0.60, kelly_scale=0.95,
    )
    assert "비중 0%" in text
    assert "비" in text and "씨" in text


def test_allocation_high_sharpe_full_betting():
    alloc = pd.Series({"A": 0.5, "B": 0.5, "CASH": 0.0})
    text = narrate_allocation(
        alloc, {}, expected_return=0.15, volatility=0.12, sharpe=1.2, kelly_scale=0.95,
    )
    assert "양호" in text or "풀 베팅" in text


def test_allocation_low_sharpe_conservative():
    alloc = pd.Series({"A": 0.2, "B": 0.1, "CASH": 0.7})
    text = narrate_allocation(
        alloc, {}, expected_return=0.02, volatility=0.20, sharpe=0.10, kelly_scale=0.30,
    )
    assert "낮" in text or "보수" in text or "현금" in text


# ---------- narrate_warnings ----------

def test_warnings_high_risky_weight():
    alloc = pd.Series({"A": 0.95, "CASH": 0.05})
    text = narrate_warnings(alloc, volatility=0.12)
    assert "위험자산 비중" in text


def test_warnings_high_volatility():
    alloc = pd.Series({"A": 0.5, "CASH": 0.5})
    text = narrate_warnings(alloc, volatility=0.25)
    assert "변동성" in text


def test_warnings_always_mentions_rebalance():
    alloc = pd.Series({"A": 0.5, "CASH": 0.5})
    text = narrate_warnings(alloc, volatility=0.10)
    assert "리밸런싱" in text


# ---------- narrate_all (통합) ----------

def test_narrate_all_includes_all_sections(sample_scores):
    text = narrate_all(
        regime_label="risk_on", regime_score=0.4,
        regime_contributions={"vix": 0.2, "hy_oas": 0.2},
        regime_feature_values={"vix": 15.0, "hy_oas": 2.8},
        scores=sample_scores,
        allocation=pd.Series({"A": 0.5, "B": 0.3, "CASH": 0.2}),
        asset_names={"A": "에이", "B": "비"},
        expected_return=0.08, volatility=0.12, sharpe=0.55, kelly_scale=0.80,
    )
    # 네 섹션 제목이 모두 포함
    assert "시장 진단" in text
    assert "자산별 평가" in text
    assert "비중 해석" in text
    assert "위험 신호" in text


def test_narrate_all_handles_missing_regime_features():
    """regime_feature_values 가 일부 누락되어도 OK."""
    text = narrate_all(
        regime_label="neutral", regime_score=0.05,
        regime_contributions={"vix": 0.05},
        regime_feature_values={},  # 비어있어도 안 터짐
        scores=pd.DataFrame({"technical": [50], "regime": [50], "composite": [50]},
                            index=["X"]),
        allocation=pd.Series({"X": 0.5, "CASH": 0.5}),
        asset_names={"X": "엑스"},
        expected_return=0.04, volatility=0.10, sharpe=0.30, kelly_scale=0.50,
    )
    assert len(text) > 0


def test_feature_labels_have_korean():
    """모든 feature 의 한국어 라벨이 정의되어 있음."""
    expected_features = {"vix", "hy_oas", "spread_10_2", "usd_krw", "base_rate"}
    assert expected_features.issubset(set(FEATURE_LABELS.keys()))
