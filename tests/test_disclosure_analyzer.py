"""
LLM 공시 분석 테스트 — 실제 Claude API 호출 없이 mock 으로 검증.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.data import cache
from src.llm import disclosure_analyzer


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def reset_yaml_cache():
    from src.utils.config_loader import load_yaml
    load_yaml.cache_clear()


class FakeContentBlock:
    def __init__(self, text: str):
        self.text = text


class FakeMessage:
    def __init__(self, text: str, in_tokens: int = 100, out_tokens: int = 50):
        self.content = [FakeContentBlock(text)]
        self.usage = SimpleNamespace(input_tokens=in_tokens, output_tokens=out_tokens)


class FakeClient:
    """Anthropic 클라이언트를 흉내."""

    def __init__(self, response_text: str):
        self.messages = self
        self._response = response_text
        self.last_request: dict | None = None

    def create(self, model, max_tokens, temperature, messages):
        self.last_request = {
            "model": model, "max_tokens": max_tokens,
            "temperature": temperature, "messages": messages,
        }
        return FakeMessage(self._response)


# ---------- _parse_response ----------

def test_parse_response_raw_json():
    text = '{"summary": "OK", "sentiment_score": 0.5}'
    out = disclosure_analyzer._parse_response(text)
    assert out["summary"] == "OK"
    assert out["sentiment_score"] == 0.5


def test_parse_response_markdown_fenced():
    text = "음, 분석 결과는:\n```json\n{\"x\": 1}\n```\n끝."
    out = disclosure_analyzer._parse_response(text)
    assert out == {"x": 1}


def test_parse_response_finds_braces():
    text = "전문: 어쩌고 {\"a\": 1, \"b\": 2} 추가 텍스트"
    out = disclosure_analyzer._parse_response(text)
    assert out == {"a": 1, "b": 2}


# ---------- _validate_and_clamp ----------

def test_validate_clamps_score_above_bounds():
    parsed = {
        "summary": "s", "sentiment_score": 5.0, "sentiment_label": "호재",
        "keywords": ["k"], "reasoning": "r",
    }
    bounds = {"min": -1.0, "max": 1.0}
    out = disclosure_analyzer._validate_and_clamp(parsed, bounds)
    assert out["sentiment_score"] == 1.0


def test_validate_clamps_score_below_bounds():
    parsed = {
        "summary": "s", "sentiment_score": -3.0, "sentiment_label": "악재",
        "keywords": [], "reasoning": "r",
    }
    bounds = {"min": -1.0, "max": 1.0}
    out = disclosure_analyzer._validate_and_clamp(parsed, bounds)
    assert out["sentiment_score"] == -1.0


def test_validate_missing_field_raises():
    parsed = {"summary": "s"}
    with pytest.raises(ValueError, match="누락"):
        disclosure_analyzer._validate_and_clamp(parsed, {"min": -1, "max": 1})


# ---------- API key handling ----------

def test_missing_anthropic_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(disclosure_analyzer.AnthropicKeyMissingError):
        disclosure_analyzer._get_api_key()


# ---------- analyze_disclosure with mock client ----------

def test_analyze_disclosure_happy_path(isolated_cache):
    fake_response = """\
{
  "summary": "삼성전자 2024년 매출 300조 돌파, 전년비 +15%",
  "sentiment_score": 0.7,
  "sentiment_label": "호재",
  "keywords": ["매출 성장", "반도체 회복", "AI 수요"],
  "reasoning": "원문에서 '매출 300조원, 전년비 15% 증가' 명시 — 강한 호재"
}"""
    client = FakeClient(fake_response)
    result = disclosure_analyzer.analyze_disclosure(
        corp_name="삼성전자", stock_code="005930",
        report_name="사업보고서", report_date="2025-03-15",
        content="(공시 원문)",
        use_cache=False,
        client=client,
    )
    assert result.sentiment_score == pytest.approx(0.7)
    assert result.sentiment_label == "호재"
    assert len(result.keywords) == 3
    assert "매출 성장" in result.keywords
    assert result.input_tokens == 100
    assert result.output_tokens == 50


def test_analyze_disclosure_uses_cache(isolated_cache):
    """두 번째 호출은 client 사용 X."""
    fake_response = """{"summary": "s", "sentiment_score": 0.3, "sentiment_label": "호재", "keywords": ["k"], "reasoning": "r"}"""
    client = FakeClient(fake_response)
    args = dict(
        corp_name="X", stock_code="000000", report_name="N",
        report_date="2025-01-01", content="abc",
    )
    r1 = disclosure_analyzer.analyze_disclosure(**args, use_cache=True, client=client)

    # 호출 카운트 — 같은 입력이면 client 호출 안 됨
    client.last_request = None
    r2 = disclosure_analyzer.analyze_disclosure(**args, use_cache=True, client=client)
    assert client.last_request is None  # 캐시에서 가져옴
    assert r2.sentiment_score == r1.sentiment_score


def test_analyze_disclosure_clamps_out_of_bounds(isolated_cache):
    fake_response = """{"summary": "s", "sentiment_score": 99, "sentiment_label": "호재", "keywords": [], "reasoning": "r"}"""
    client = FakeClient(fake_response)
    result = disclosure_analyzer.analyze_disclosure(
        corp_name="X", stock_code="000001", report_name="N",
        report_date="2025-01-01", content="x",
        use_cache=False, client=client,
    )
    assert result.sentiment_score == 1.0


def test_analyze_disclosure_invalid_json_raises(isolated_cache):
    client = FakeClient("이건 JSON 이 아닙니다")
    with pytest.raises(Exception):  # JSONDecodeError 또는 ValueError
        disclosure_analyzer.analyze_disclosure(
            corp_name="X", stock_code="000002", report_name="N",
            report_date="2025-01-01", content="x",
            use_cache=False, client=client,
        )


# ---------- 비용 추정 ----------

def test_estimate_cost_known_model():
    cost = disclosure_analyzer.estimate_cost_usd(
        input_tokens=1_000_000, output_tokens=1_000_000,
        model="claude-haiku-4-5",
    )
    # Haiku: 0.25 + 1.25 = 1.50 USD per 1M+1M tokens
    assert cost == pytest.approx(1.50)


def test_estimate_cost_unknown_model_zero():
    cost = disclosure_analyzer.estimate_cost_usd(
        input_tokens=1000, output_tokens=1000,
        model="unknown-model-x",
    )
    assert cost == 0.0
