"""
일일 리포트 해설(narrative) 생성기 — Phase 4 보조.

데이터 표만 늘어놓으면 한눈에 안 들어오니, **사람이 읽는 글** 로 풀어 설명합니다.
모든 규칙은 결정론적 (LLM 없이도 동일 입력 → 동일 출력) — 백테스트·재현성에 유리.

추후 ANTHROPIC_API_KEY 가 있으면 이 출력을 Claude 가 다시 다듬는
"polish layer" 를 src/llm/ 에 얹을 수 있음.
"""
from __future__ import annotations

from typing import Mapping

import pandas as pd


# ----------------------------------------------------------------------
# 지표 해석 헬퍼 (역사적 정상 범위 기준)
# ----------------------------------------------------------------------
def _interpret_vix(v: float) -> str:
    if pd.isna(v):
        return "데이터 없음"
    if v < 12:
        return f"VIX `{v:.1f}` — 매우 낮음 (자만 우려)"
    if v < 20:
        return f"VIX `{v:.1f}` — 평년 평균 미만 (위험선호 강함)"
    if v < 30:
        return f"VIX `{v:.1f}` — 평년 평균 초과 (긴장 시작)"
    return f"VIX `{v:.1f}` — 매우 높음 (패닉 신호)"


def _interpret_oas(v: float) -> str:
    if pd.isna(v):
        return "데이터 없음"
    if v < 3:
        return f"하이일드 OAS `{v:.2f}%` — 좁음 (신용 안정)"
    if v < 5:
        return f"하이일드 OAS `{v:.2f}%` — 정상 범위"
    if v < 7:
        return f"하이일드 OAS `{v:.2f}%` — 확대 (긴장)"
    return f"하이일드 OAS `{v:.2f}%` — 매우 확대 (위기 신호)"


def _interpret_spread(v: float) -> str:
    if pd.isna(v):
        return "데이터 없음"
    if v < -0.2:
        return f"10Y-2Y `{v:+.2f}%p` — 역전 (경기 우려)"
    if v < 0.5:
        return f"10Y-2Y `{v:+.2f}%p` — 평평 (애매)"
    if v < 1.5:
        return f"10Y-2Y `{v:+.2f}%p` — 정상 우상향"
    return f"10Y-2Y `{v:+.2f}%p` — 가파른 우상향 (회복 기대)"


def _interpret_usdkrw(v: float) -> str:
    if pd.isna(v):
        return "데이터 없음"
    if v < 1200:
        return f"원/달러 `{v:,.0f}원` — 강원 (위험자산 우호)"
    if v < 1350:
        return f"원/달러 `{v:,.0f}원` — 평년 수준"
    if v < 1450:
        return f"원/달러 `{v:,.0f}원` — 약원 (긴장)"
    return f"원/달러 `{v:,.0f}원` — 매우 약원 (위험회피)"


def _interpret_base_rate(v: float) -> str:
    if pd.isna(v):
        return "데이터 없음"
    if v < 1.5:
        return f"한국 기준금리 `{v:.2f}%` — 매우 저금리 (완화)"
    if v < 3:
        return f"한국 기준금리 `{v:.2f}%` — 저금리"
    if v < 4:
        return f"한국 기준금리 `{v:.2f}%` — 중립~긴축"
    return f"한국 기준금리 `{v:.2f}%` — 고금리 (긴축)"


FEATURE_LABELS = {
    "vix": "VIX (공포지수)",
    "hy_oas": "하이일드 OAS (신용 스프레드)",
    "spread_10_2": "10Y-2Y 장단기 스프레드",
    "usd_krw": "원/달러 환율",
    "base_rate": "한국 기준금리",
}

FEATURE_INTERPRETERS = {
    "vix": _interpret_vix,
    "hy_oas": _interpret_oas,
    "spread_10_2": _interpret_spread,
    "usd_krw": _interpret_usdkrw,
    "base_rate": _interpret_base_rate,
}


def _feature_label(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


def _interpret_feature(name: str, value: float) -> str:
    fn = FEATURE_INTERPRETERS.get(name)
    return fn(value) if fn else f"{name} = {value}"


# ----------------------------------------------------------------------
# 1) 시장 국면 해설
# ----------------------------------------------------------------------
def narrate_regime(
    label: str,
    score: float,
    contributions: Mapping[str, float],
    feature_values: Mapping[str, float],
) -> str:
    label_kr = {
        "risk_on": "위험선호 (Risk-On)",
        "risk_off": "위험회피 (Risk-Off)",
        "neutral": "중립 (Neutral)",
    }.get(label, label)

    # 강도 표현
    abs_score = abs(score) if pd.notna(score) else 0
    if abs_score > 0.6:
        strength = "강한"
    elif abs_score > 0.3:
        strength = "뚜렷한"
    elif abs_score > 0.1:
        strength = "약한"
    else:
        strength = "거의 중립에 가까운"

    lines: list[str] = []
    lines.append(
        f"오늘 시장은 **{strength} {label_kr}** 국면입니다 "
        f"(종합 점수 `{score:+.2f}`, 범위 -1 ↔ +1)."
    )

    # 가장 큰 기여 지표
    sorted_contrib = sorted(
        ((n, c) for n, c in contributions.items() if pd.notna(c)),
        key=lambda x: -abs(x[1]),
    )
    if sorted_contrib:
        top_name, top_contrib = sorted_contrib[0]
        top_value = feature_values.get(top_name, float("nan"))
        direction = "위험선호" if top_contrib > 0 else "위험회피"
        lines.append(
            f"가장 큰 영향: **{_feature_label(top_name)}** — "
            f"{_interpret_feature(top_name, top_value)}. "
            f"이 한 지표만으로 종합 점수에 `{top_contrib:+.2f}` ({direction} 방향) 기여했어요."
        )

    # 상반된 신호 (강한 risk-on 환경에서도 risk-off 신호가 있을 수 있음)
    counter_signals = [
        (n, c) for n, c in contributions.items()
        if pd.notna(c) and ((score > 0 and c < -0.05) or (score < 0 and c > 0.05))
    ]
    if counter_signals:
        counter_signals.sort(key=lambda x: -abs(x[1]))
        cn, cc = counter_signals[0]
        cv = feature_values.get(cn, float("nan"))
        lines.append(
            f"단, **{_feature_label(cn)}** 는 반대 방향 신호를 보내고 있어요 "
            f"({_interpret_feature(cn, cv)}). 시장이 한 방향으로만 흐르고 있는 건 아닙니다."
        )

    return "\n\n".join(lines)


# ----------------------------------------------------------------------
# 2) 자산 점수 해설
# ----------------------------------------------------------------------
def narrate_scores(
    scores: pd.DataFrame,
    asset_names: Mapping[str, str],
) -> str:
    if scores.empty or "composite" not in scores.columns:
        return "_점수 데이터 없음._"

    sorted_scores = scores.sort_values("composite", ascending=False)
    top = sorted_scores.iloc[0]
    bottom = sorted_scores.iloc[-1]
    top_code, bottom_code = sorted_scores.index[0], sorted_scores.index[-1]
    top_name = asset_names.get(top_code, top_code)
    bottom_name = asset_names.get(bottom_code, bottom_code)

    lines: list[str] = []

    # 최고 자산 — 이유 설명
    top_reasons: list[str] = []
    if top.get("technical", 50) >= 75:
        top_reasons.append(f"기술적 모멘텀이 강합니다 (`{top['technical']:.1f}점`, 최근 ~3개월 가격 추세 상위)")
    if "fundamental" in scores.columns and pd.notna(top.get("fundamental")) and top["fundamental"] >= 70:
        top_reasons.append(f"펀더멘털도 우수 (`{top['fundamental']:.1f}점`, PER/PBR/ROE 등 종합)")
    if top.get("regime", 50) >= 65:
        top_reasons.append(f"현재 시장 국면에 적합 (`{top['regime']:.1f}점`)")
    if not top_reasons:
        top_reasons.append("개별 신호는 평범하지만 합성하면 가장 균형이 좋음")
    lines.append(
        f"가장 매력적인 자산은 **{top_name}** — 종합 `{top['composite']:.1f}점`. "
        + " 그리고 ".join(top_reasons) + "."
    )

    # 최저 자산
    bottom_reasons: list[str] = []
    if bottom.get("technical", 50) <= 30:
        bottom_reasons.append(f"기술적 모멘텀이 약합니다 (`{bottom['technical']:.1f}점`, 최근 가격 부진)")
    if "fundamental" in scores.columns and pd.notna(bottom.get("fundamental")) and bottom["fundamental"] <= 30:
        bottom_reasons.append(f"펀더멘털도 약함 (`{bottom['fundamental']:.1f}점`)")
    if bottom.get("regime", 50) <= 40:
        bottom_reasons.append(f"현재 국면과 안 맞습니다 (`{bottom['regime']:.1f}점`)")
    if not bottom_reasons:
        bottom_reasons.append("특별히 나쁜 점은 없지만 다른 자산이 더 매력적")
    lines.append(
        f"반대로 가장 약한 자산은 **{bottom_name}** — 종합 `{bottom['composite']:.1f}점`. "
        + " 그리고 ".join(bottom_reasons) + "."
    )

    # 분포 요약
    high = (sorted_scores["composite"] >= 60).sum()
    low = (sorted_scores["composite"] < 40).sum()
    lines.append(
        f"전체 {len(sorted_scores)}개 자산 중 종합 60점 이상이 **{high}개**, "
        f"40점 미만이 **{low}개** 입니다."
    )
    return "\n\n".join(lines)


# ----------------------------------------------------------------------
# 3) 비중 해설
# ----------------------------------------------------------------------
def narrate_allocation(
    allocation: pd.Series,
    asset_names: Mapping[str, str],
    expected_return: float,
    volatility: float,
    sharpe: float,
    kelly_scale: float,
) -> str:
    cash_weight = float(allocation.get("CASH", 0.0))
    risky_weight = 1.0 - cash_weight

    # 위험자산 톱2
    risky = allocation.drop("CASH", errors="ignore").sort_values(ascending=False)
    risky = risky[risky > 0]
    top_two = risky.head(2)
    zero_weight = (allocation.drop("CASH", errors="ignore") == 0).sum()

    lines: list[str] = []

    # 핵심 베팅 요약
    if len(top_two) > 0:
        top_names = " + ".join(
            f"**{asset_names.get(c, c)}** (`{w * 100:.1f}%`)" for c, w in top_two.items()
        )
        lines.append(
            f"핵심 베팅은 {top_names} — 합산 `{top_two.sum() * 100:.1f}%` 입니다."
        )

    # 비중 0 자산
    if zero_weight > 0:
        zero_codes = allocation.drop("CASH", errors="ignore")
        zero_codes = zero_codes[zero_codes == 0].index.tolist()
        zero_names = ", ".join(asset_names.get(c, c) for c in zero_codes)
        lines.append(
            f"비중 0%인 자산: {zero_names} — 점수가 낮거나 최적화 결과 제외되었습니다."
        )

    # Sharpe 해석
    if sharpe is not None and pd.notna(sharpe):
        if sharpe >= 1.0:
            sharpe_comment = "매우 양호한 위험조정 수익 — 풀 베팅에 가깝게 추천"
        elif sharpe >= 0.5:
            sharpe_comment = "보통 수준의 위험조정 수익 — 적정 비중"
        elif sharpe >= 0.2:
            sharpe_comment = "낮은 위험조정 수익 — 보수적으로"
        else:
            sharpe_comment = "위험 대비 보상이 낮음 — 현금 비중 확대"
        lines.append(
            f"포트폴리오 Sharpe `{sharpe:.2f}` ({sharpe_comment}). "
            f"기대수익률 `{expected_return * 100:+.2f}%` / 변동성 `{volatility * 100:.2f}%`."
        )

    # Kelly 해석
    if kelly_scale is not None and pd.notna(kelly_scale):
        if kelly_scale >= 0.85:
            kelly_comment = "Half-Kelly 결과 거의 풀 베팅에 가깝게 — 시장 환경과 신호가 우호적"
        elif kelly_scale >= 0.5:
            kelly_comment = "Half-Kelly 결과 위험자산 절반 이상 — 균형 잡힌 베팅"
        elif kelly_scale >= 0.2:
            kelly_comment = "Half-Kelly 결과 조심스러운 비중 — 현금 비중 큼"
        else:
            kelly_comment = "Half-Kelly 결과 매우 보수적 — 사실상 현금 위주"
        lines.append(
            f"현금 `{cash_weight * 100:.1f}%`, 위험자산 `{risky_weight * 100:.1f}%`. "
            f"{kelly_comment}."
        )

    return "\n\n".join(lines)


# ----------------------------------------------------------------------
# 4) 주의 사항 해설
# ----------------------------------------------------------------------
def narrate_warnings(
    allocation: pd.Series,
    volatility: float,
) -> str:
    risky_weight = 1.0 - float(allocation.get("CASH", 0.0))
    lines: list[str] = []

    if risky_weight >= 0.9:
        lines.append(
            "💡 위험자산 비중이 매우 높습니다. 시장 국면이 갑자기 바뀌면 다음 리밸런싱(보통 월말) "
            "전까지 손실이 클 수 있어요."
        )

    if volatility is not None and pd.notna(volatility) and volatility >= 0.20:
        lines.append(
            f"📉 추천 포트폴리오의 연환산 변동성이 `{volatility * 100:.0f}%` — 평년보다 큰 편. "
            f"백테스트 기준 MDD 가 -15~25% 수준까지 갈 수 있어요."
        )

    lines.append(
        "🔄 다음 리밸런싱(보통 월말)까지 이 비중을 유지하는 게 기본 전제예요. "
        "중간에 시장이 크게 흔들리면 임의 비중 조정보다는 다음 일일 리포트를 기다리는 게 일관성에 좋아요."
    )

    return "\n\n".join(f"- {ln}" for ln in lines)


# ----------------------------------------------------------------------
# 통합
# ----------------------------------------------------------------------
def narrate_all(
    regime_label: str,
    regime_score: float,
    regime_contributions: Mapping[str, float],
    regime_feature_values: Mapping[str, float],
    scores: pd.DataFrame,
    allocation: pd.Series,
    asset_names: Mapping[str, str],
    expected_return: float,
    volatility: float,
    sharpe: float,
    kelly_scale: float,
) -> str:
    """모든 해설을 하나의 Markdown 문자열로."""
    parts = [
        "### 🌐 시장 진단",
        narrate_regime(regime_label, regime_score, regime_contributions, regime_feature_values),
        "",
        "### 🎯 자산별 평가",
        narrate_scores(scores, asset_names),
        "",
        "### 💼 비중 해석",
        narrate_allocation(
            allocation, asset_names, expected_return, volatility, sharpe, kelly_scale
        ),
        "",
        "### ⚠️ 위험 신호",
        narrate_warnings(allocation, volatility),
    ]
    return "\n\n".join(parts)
