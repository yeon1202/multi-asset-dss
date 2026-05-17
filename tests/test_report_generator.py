"""
리포트 생성기 테스트.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.report.generator import (
    DISCLAIMER,
    ReportContext,
    render,
    write_report,
)


@pytest.fixture
def sample_ctx() -> ReportContext:
    scores = pd.DataFrame({
        "technical": [80, 60, 40],
        "regime":    [70, 50, 30],
        "composite": [75, 55, 35],
    }, index=["069500", "148070", "132030"])
    alloc = pd.Series({
        "069500": 0.35,
        "148070": 0.20,
        "132030": 0.10,
        "CASH": 0.35,
    })
    return ReportContext(
        as_of=date(2024, 1, 15),
        regime_label="risk_on",
        regime_score=0.45,
        regime_contributions={"vix": 0.20, "hy_oas": 0.15, "spread_10_2": 0.10},
        scores=scores,
        allocation=alloc,
        asset_names={"069500": "KODEX 200", "148070": "KOSEF 국고채10년",
                     "132030": "KODEX 골드"},
        expected_return=0.08,
        volatility=0.15,
        sharpe=0.55,
        kelly_fraction=0.65,
    )


def test_render_includes_disclaimer(sample_ctx):
    text = render(sample_ctx)
    assert DISCLAIMER in text


def test_render_includes_date(sample_ctx):
    text = render(sample_ctx)
    assert "2024-01-15" in text


def test_render_includes_regime_label_kr(sample_ctx):
    text = render(sample_ctx)
    assert "RISK-ON" in text
    assert "🟢" in text


def test_render_shows_asset_names(sample_ctx):
    text = render(sample_ctx)
    assert "KODEX 200" in text
    assert "KOSEF 국고채10년" in text
    assert "KODEX 골드" in text
    assert "💵 현금" in text


def test_render_shows_allocation_percentages(sample_ctx):
    text = render(sample_ctx)
    # 비중이 % 로 표시
    assert "35.0%" in text  # KODEX 200 또는 CASH
    assert "20.0%" in text
    assert "10.0%" in text


def test_render_no_prev_no_diff_section(sample_ctx):
    text = render(sample_ctx)
    assert "전일 대비 변경" not in text


def test_render_with_prev_shows_diff(sample_ctx):
    prev = pd.Series({"069500": 0.20, "148070": 0.30, "132030": 0.05, "CASH": 0.45})
    ctx = ReportContext(**{**sample_ctx.__dict__, "prev_allocation": prev})
    text = render(ctx)
    assert "전일 대비 변경" in text
    # KODEX 200: 20 → 35 (+15%p) 가 표시
    assert "+15.0%p" in text


def test_render_diff_threshold(sample_ctx):
    """0.5%p 미만 변화는 표시 안 함."""
    prev = pd.Series({"069500": 0.348, "148070": 0.20, "132030": 0.10, "CASH": 0.352})
    ctx = ReportContext(**{**sample_ctx.__dict__, "prev_allocation": prev})
    text = render(ctx)
    # 069500 변동 0.2%p < 0.5%p → "변동 없음" 메시지
    assert "변동 없음" in text


def test_write_report_creates_file(sample_ctx, tmp_path):
    path = write_report(sample_ctx, output_dir=tmp_path)
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "📊 일일 포트폴리오 리포트" in content
    assert DISCLAIMER in content


def test_render_sections_order(sample_ctx):
    text = render(sample_ctx)
    # 면책 → 시장 국면 → 자산별 점수 → 추천 포트폴리오 → 주의사항
    idx_disclaimer = text.index(DISCLAIMER)
    idx_regime = text.index("시장 국면")
    idx_scores = text.index("자산별 점수")
    idx_alloc = text.index("추천 포트폴리오")
    idx_warn = text.index("주의사항")
    assert idx_disclaimer < idx_regime < idx_scores < idx_alloc < idx_warn
