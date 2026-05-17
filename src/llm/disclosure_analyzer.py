"""
DART 공시 정성 분석 — Phase 6.

Claude API 로 공시 원문을 받아:
  1. 한 줄 요약
  2. 호재/악재 점수 (-1 ~ +1)
  3. 핵심 키워드 3개
  4. 분석 근거

PROJECT_SPEC.md §9.2:
  - 모든 LLM 출력은 원본 자료와 교차 검증
  - 환각 방지 위해 항상 원문 인용 요구
  - 비용 모니터링 필수
  - LLM 출력만으로 추천 결정 X
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from loguru import logger

from src.data import cache
from src.utils.config_loader import PROJECT_ROOT, load_llm_config

load_dotenv(PROJECT_ROOT / ".env")


class AnthropicKeyMissingError(RuntimeError):
    """ANTHROPIC_API_KEY 가 환경변수에 없을 때."""


@dataclass(frozen=True)
class DisclosureAnalysis:
    """공시 1건의 분석 결과."""

    summary: str            # 한 줄 요약
    sentiment_score: float  # -1.0 ~ +1.0
    sentiment_label: str    # "호재" | "악재" | "중립"
    keywords: list[str]     # 핵심 키워드
    reasoning: str          # 분석 근거
    model: str              # 사용된 모델
    input_tokens: int       # 비용 추적
    output_tokens: int


def _get_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise AnthropicKeyMissingError(
            "ANTHROPIC_API_KEY 가 설정되지 않았습니다.\n"
            "1) https://console.anthropic.com 에서 API 키 발급\n"
            f"2) {PROJECT_ROOT}/.env 파일에 'ANTHROPIC_API_KEY=발급받은키' 추가"
        )
    return key


PROMPT_TEMPLATE = """\
당신은 한국 주식시장 전문 분석가입니다. 아래 DART 공시 원문을 분석해주세요.

## 공시 정보
- 회사: {corp_name}
- 종목코드: {stock_code}
- 공시명: {report_name}
- 공시일자: {report_date}

## 공시 원문
{content}

## 분석 지시
다음 JSON 형식으로만 응답하세요. 추가 설명 금지.

{{
  "summary": "한 줄 요약 (한국어, 100자 이내)",
  "sentiment_score": -1.0 ~ +1.0 사이 실수 (-1=극단적 악재, 0=중립, +1=극단적 호재),
  "sentiment_label": "호재" | "악재" | "중립",
  "keywords": ["키워드1", "키워드2", "키워드3"],
  "reasoning": "왜 그 점수인지 근거 (원문 인용 포함, 200자 이내)"
}}

## 주의사항
- 단정적 표현 금지 (예: "확실히 오른다" X)
- 원문에 없는 사실 추측 금지 (환각 방지)
- 회계상 일회성 손익은 sentiment_score 에 신중히 반영
"""


def _parse_response(text: str) -> dict[str, Any]:
    """Claude 응답에서 JSON 블록 추출."""
    # JSON 블록 찾기 (마크다운 ```json 또는 raw)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        # 첫 { 부터 마지막 } 까지
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
    return json.loads(text)


def _validate_and_clamp(parsed: dict[str, Any], bounds: dict[str, float]) -> dict[str, Any]:
    """파싱된 dict 의 필드 검증 + sentiment_score 범위 제한."""
    required = {"summary", "sentiment_score", "sentiment_label", "keywords", "reasoning"}
    missing = required - set(parsed.keys())
    if missing:
        raise ValueError(f"LLM 응답에 필수 필드 누락: {missing}")
    score = float(parsed["sentiment_score"])
    score = max(bounds["min"], min(bounds["max"], score))
    parsed["sentiment_score"] = score
    if not isinstance(parsed["keywords"], list):
        parsed["keywords"] = []
    return parsed


def analyze_disclosure(
    corp_name: str,
    stock_code: str,
    report_name: str,
    report_date: str,
    content: str,
    use_cache: bool = True,
    client: Any | None = None,
) -> DisclosureAnalysis:
    """
    공시 1건을 Claude 로 분석.

    `client` 를 직접 주입하면 (테스트 mock 용) API 키 없이도 동작.
    """
    cfg = load_llm_config()
    cache_key = f"llm_disclosure_{stock_code}_{report_date}_{hash(content) & 0xffffffff:08x}"

    if use_cache and cache.is_fresh(cache_key, cfg["cache_ttl_days"]):
        cached = cache.load(cache_key)
        if cached is not None and not cached.empty:
            row = cached.iloc[0]
            logger.info(f"[캐시] LLM 분석 {stock_code} {report_date}")
            return DisclosureAnalysis(
                summary=str(row["summary"]),
                sentiment_score=float(row["sentiment_score"]),
                sentiment_label=str(row["sentiment_label"]),
                keywords=str(row["keywords"]).split("|"),
                reasoning=str(row["reasoning"]),
                model=str(row["model"]),
                input_tokens=int(row["input_tokens"]),
                output_tokens=int(row["output_tokens"]),
            )

    if client is None:
        from anthropic import Anthropic
        client = Anthropic(api_key=_get_api_key())

    prompt = PROMPT_TEMPLATE.format(
        corp_name=corp_name, stock_code=stock_code,
        report_name=report_name, report_date=report_date,
        content=content[:8000],  # 토큰 제한 — 너무 길면 잘림
    )
    logger.info(f"[API] Claude 분석 {stock_code} {report_date}")
    message = client.messages.create(
        model=cfg["model"],
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        messages=[{"role": "user", "content": prompt}],
    )
    # Anthropic SDK 응답 구조: message.content[0].text
    if not message.content:
        raise RuntimeError("Claude API 가 빈 응답을 반환")
    text = "".join(getattr(b, "text", "") for b in message.content)
    parsed = _parse_response(text)
    parsed = _validate_and_clamp(parsed, cfg["sentiment_bounds"])

    result = DisclosureAnalysis(
        summary=parsed["summary"],
        sentiment_score=parsed["sentiment_score"],
        sentiment_label=parsed["sentiment_label"],
        keywords=parsed["keywords"],
        reasoning=parsed["reasoning"],
        model=cfg["model"],
        input_tokens=getattr(message.usage, "input_tokens", 0) if hasattr(message, "usage") else 0,
        output_tokens=getattr(message.usage, "output_tokens", 0) if hasattr(message, "usage") else 0,
    )

    # 캐시 저장 (DataFrame 형태)
    import pandas as pd
    row = pd.DataFrame([{
        "summary": result.summary,
        "sentiment_score": result.sentiment_score,
        "sentiment_label": result.sentiment_label,
        "keywords": "|".join(result.keywords),
        "reasoning": result.reasoning,
        "model": result.model,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }])
    cache.save(cache_key, row)
    return result


# ----------------------------------------------------------------------
# 비용 추적
# ----------------------------------------------------------------------
# Claude API 가격 (USD per 1M tokens, 2026-05 기준 — console.anthropic.com 확인)
#   Haiku:  $0.25 input / $1.25 output
#   Sonnet: $3.00 input / $15.00 output
#   Opus:   $15.00 input / $75.00 output
PRICING = {
    "claude-haiku-4-5":  (0.25, 1.25),
    "claude-haiku-4-7":  (0.25, 1.25),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-7": (3.0, 15.0),
    "claude-opus-4-5":   (15.0, 75.0),
    "claude-opus-4-6":   (15.0, 75.0),
    "claude-opus-4-7":   (15.0, 75.0),
}


def estimate_cost_usd(
    input_tokens: int,
    output_tokens: int,
    model: str,
) -> float:
    """대략적인 USD 비용 추정 (정확한 가격은 console.anthropic.com 확인)."""
    if model not in PRICING:
        return 0.0
    in_rate, out_rate = PRICING[model]
    return (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate
