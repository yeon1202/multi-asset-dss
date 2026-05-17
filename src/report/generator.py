"""
일일 리포트 생성기 — Phase 4.

Markdown 으로 저장 (reports/YYYY-MM-DD.md). 사람이 바로 읽고
필요시 PDF 변환·이메일/텔레그램 전송 가능.

PROJECT_SPEC.md §9.3 책임:
  - 첫 줄에 면책 문구
  - 모든 추천에 근거 표시
  - "확률 X%" 같은 단정 금지
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping

import pandas as pd

from src.utils.config_loader import PROJECT_ROOT


DISCLAIMER = (
    "> ⚠️ **면책**: 이 리포트는 정보 제공 목적이며, 투자 결정과 그 결과는 본인 책임입니다. "
    "자동 주문 기능은 포함되어 있지 않으며, 모든 주문은 사용자가 증권사 앱에서 직접 수행합니다."
)


@dataclass(frozen=True)
class ReportContext:
    """리포트 생성에 필요한 입력 묶음."""

    as_of: date
    regime_label: str
    regime_score: float
    regime_contributions: Mapping[str, float]
    scores: pd.DataFrame                # composite_score 산출물 (tech/regime/composite)
    allocation: pd.Series               # 자산 + CASH 비중 (sum=1)
    asset_names: Mapping[str, str]      # 종목코드 → 사람용 이름
    expected_return: float              # 위험자산 포트폴리오 연환산 기대수익률
    volatility: float                   # 연환산 변동성
    sharpe: float                       # 샤프
    kelly_fraction: float               # 실제 적용된 켈리 스케일
    prev_allocation: pd.Series | None = None  # 직전 리포트의 비중 (diff 계산용)


def _fmt_pct(x: float, sign: bool = False) -> str:
    if x is None or pd.isna(x):
        return "—"
    s = f"{x:+.1f}%" if sign else f"{x:.1f}%"
    return s


def _regime_emoji(label: str) -> str:
    return {"risk_on": "🟢", "risk_off": "🔴", "neutral": "⚪"}.get(label, "❓")


def _label_kr(label: str) -> str:
    return {
        "risk_on": "RISK-ON (위험선호)",
        "risk_off": "RISK-OFF (위험회피)",
        "neutral": "NEUTRAL (중립)",
    }.get(label, label)


def render(ctx: ReportContext) -> str:
    """ReportContext 를 Markdown 문자열로 렌더링."""
    lines: list[str] = []

    lines.append(f"# 📊 일일 포트폴리오 리포트 — {ctx.as_of}")
    lines.append("")
    lines.append(DISCLAIMER)
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1) 시장 국면
    lines.append("## 🌐 시장 국면")
    lines.append("")
    lines.append(
        f"- **현재 레짐**: {_regime_emoji(ctx.regime_label)} **{_label_kr(ctx.regime_label)}**"
    )
    lines.append(f"- **종합 점수**: `{ctx.regime_score:+.3f}` (-1 ↔ +1)")
    lines.append("")
    if ctx.regime_contributions:
        lines.append("주요 기여 지표:")
        sorted_contrib = sorted(
            ctx.regime_contributions.items(), key=lambda x: -abs(x[1])
        )
        for feat, contrib in sorted_contrib[:5]:
            lines.append(f"  - `{feat}`: {contrib:+.3f}")
        lines.append("")

    # 2) 자산별 점수
    lines.append("## 🎯 자산별 점수")
    lines.append("")
    lines.append("| 종목 | 기술적 (모멘텀) | 레짐 적합도 | **종합 점수** |")
    lines.append("|---|---|---|---|")
    scores_sorted = ctx.scores.sort_values("composite", ascending=False)
    for code, row in scores_sorted.iterrows():
        name = ctx.asset_names.get(code, code)
        lines.append(
            f"| {name} ({code}) | {row['technical']:.1f} | "
            f"{row['regime']:.1f} | **{row['composite']:.1f}** |"
        )
    lines.append("")

    # 3) 추천 비중
    lines.append("## 💼 추천 포트폴리오")
    lines.append("")
    lines.append(f"- 위험자산 포트폴리오 연환산 기대수익률: `{_fmt_pct(ctx.expected_return*100, sign=True)}`")
    lines.append(f"- 연환산 변동성: `{_fmt_pct(ctx.volatility*100)}`")
    lines.append(f"- 샤프 비율: `{ctx.sharpe:.2f}`")
    lines.append(f"- 적용된 Half-Kelly 스케일: `{ctx.kelly_fraction:.2%}`")
    lines.append("")
    lines.append("| 자산 | 비중 |")
    lines.append("|---|---|")
    alloc_sorted = ctx.allocation.sort_values(ascending=False)
    for code, w in alloc_sorted.items():
        if code == "CASH":
            label = "💵 현금"
        else:
            label = ctx.asset_names.get(code, code)
            label = f"{label} ({code})"
        lines.append(f"| {label} | **{w*100:.1f}%** |")
    lines.append("")

    # 4) 전일 대비 변경
    if ctx.prev_allocation is not None and not ctx.prev_allocation.empty:
        lines.append("## 🔄 전일 대비 변경")
        lines.append("")
        all_keys = sorted(set(ctx.allocation.index) | set(ctx.prev_allocation.index))
        changed = []
        for key in all_keys:
            today = ctx.allocation.get(key, 0.0)
            yest = ctx.prev_allocation.get(key, 0.0)
            diff = today - yest
            if abs(diff) >= 0.005:  # 0.5%p 이상만 표시
                changed.append((key, yest, today, diff))
        if changed:
            lines.append("| 자산 | 전일 | 오늘 | 변동 |")
            lines.append("|---|---|---|---|")
            for key, yest, today, diff in changed:
                if key == "CASH":
                    lbl = "💵 현금"
                else:
                    lbl = f"{ctx.asset_names.get(key, key)} ({key})"
                lines.append(
                    f"| {lbl} | {yest*100:.1f}% | {today*100:.1f}% | "
                    f"**{diff*100:+.1f}%p** |"
                )
            lines.append("")
        else:
            lines.append("_변동 없음 (0.5%p 미만 변화 무시)_")
            lines.append("")

    # 5) 주의사항
    lines.append("## ⚠️ 주의사항")
    lines.append("")
    lines.append(
        "- 본 추천은 **마코위츠 평균-분산 + Half-Kelly + 룰 기반 레짐 + 모멘텀** 의 결합입니다."
    )
    lines.append(
        "- 모든 임계값·가중치는 `config/*.yaml` 에 분리되어 있으며, "
        "백테스팅(Phase 5) 으로 검증되기 전까지는 **휴리스틱** 입니다."
    )
    lines.append(
        "- 백테스팅 미반영 — 과적합·생존자편향 가능성 존재. 결과를 절대 단정으로 받아들이지 마세요."
    )
    lines.append("")

    return "\n".join(lines)


def write_report(
    ctx: ReportContext,
    output_dir: str | Path = "reports",
) -> Path:
    """리포트를 reports/YYYY-MM-DD.md 로 저장 후 경로 반환."""
    out_dir = PROJECT_ROOT / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ctx.as_of}.md"
    path.write_text(render(ctx), encoding="utf-8")
    return path
