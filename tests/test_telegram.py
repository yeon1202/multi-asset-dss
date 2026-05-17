"""
텔레그램 알림 테스트 — HTTP 호출은 mock 으로 차단.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.notifications import telegram


@pytest.fixture(autouse=True)
def reset_yaml_cache():
    from src.utils.config_loader import load_yaml
    load_yaml.cache_clear()


# ---------- _split_message ----------

def test_split_short_message_no_split():
    out = telegram._split_message("hello", max_len=4000)
    assert out == ["hello"]


def test_split_long_message_by_lines():
    text = "\n".join([f"line {i}" for i in range(1000)])
    parts = telegram._split_message(text, max_len=200)
    assert len(parts) > 1
    # 각 조각이 max_len 안에 들어가야
    assert all(len(p) <= 200 for p in parts)
    # 합치면 원본과 동일 (개행 손실 없음)
    assert "".join(parts) == text


def test_split_preserves_content():
    text = "line A\nline B\nline C"
    parts = telegram._split_message(text, max_len=10)
    assert "".join(parts) == text


# ---------- _get_credentials ----------

def test_get_credentials_missing_raises(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    with pytest.raises(telegram.TelegramKeysMissingError):
        telegram._get_credentials()


def test_get_credentials_partial_raises(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    with pytest.raises(telegram.TelegramKeysMissingError):
        telegram._get_credentials()


def test_get_credentials_both_present(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "1234")
    token, chat = telegram._get_credentials()
    assert token == "tok" and chat == "1234"


# ---------- send_message ----------

class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class FakeSession:
    """텔레그램 호출 캡처용 가짜 세션."""

    def __init__(self, ok=True):
        self.posts: list[dict] = []
        self.ok = ok

    def post(self, url, json=None, timeout=None):
        self.posts.append({"url": url, "json": json})
        return FakeResponse({"ok": self.ok, "result": {"message_id": 1}}, 200)


def test_send_message_single_short():
    session = FakeSession()
    out = telegram.send_message(
        "hello world", token="tok", chat_id="cid", session=session,
    )
    assert len(session.posts) == 1
    assert session.posts[0]["json"]["text"] == "hello world"
    assert session.posts[0]["json"]["chat_id"] == "cid"
    # Telegram URL 패턴: https://api.telegram.org/bot<TOKEN>/sendMessage
    assert "/bottok/sendMessage" in session.posts[0]["url"]
    assert len(out) == 1


def test_send_message_long_split():
    """긴 메시지는 여러 번 POST. 기본 max_len=4000자 기준."""
    session = FakeSession()
    # 약 6000자 — 한 번에 못 보냄
    long_text = "\n".join([f"line {i} " + "x" * 100 for i in range(60)])
    assert len(long_text) > 4000  # sanity
    telegram.send_message(
        long_text, token="tok", chat_id="cid", session=session,
    )
    assert len(session.posts) > 1


def test_send_message_api_failure_raises():
    session = FakeSession(ok=False)
    with pytest.raises(RuntimeError, match="Telegram API"):
        telegram.send_message(
            "x", token="tok", chat_id="cid", session=session,
        )


def test_send_message_disable_preview():
    session = FakeSession()
    telegram.send_message(
        "url-test", token="tok", chat_id="cid", session=session,
    )
    assert session.posts[0]["json"]["disable_web_page_preview"] is True


# ---------- send_report_summary ----------

def test_send_report_summary_format():
    session = FakeSession()
    telegram.send_report_summary(
        as_of="2025-06-01",
        regime_label="risk_on",
        regime_score=0.42,
        allocation={"069500": 0.30, "148070": 0.20, "CASH": 0.50},
        asset_names={"069500": "KODEX 200", "148070": "KOSEF 국고채10년"},
        expected_return=0.08,
        sharpe=0.65,
        token="tok", chat_id="cid", session=session,
    )
    text = session.posts[0]["json"]["text"]
    assert "🟢" in text
    assert "RISK-ON" in text
    assert "KODEX 200" in text
    assert "💵 현금" in text
    assert "50.0%" in text
    assert "정보 제공 목적" in text


def test_send_report_summary_with_big_changes():
    session = FakeSession()
    telegram.send_report_summary(
        as_of="2025-06-01",
        regime_label="risk_off",
        regime_score=-0.4,
        allocation={"069500": 0.10, "CASH": 0.90},
        asset_names={"069500": "KODEX 200"},
        big_changes={"069500": -0.20, "CASH": 0.20},
        token="tok", chat_id="cid", session=session,
    )
    text = session.posts[0]["json"]["text"]
    assert "큰 변화" in text
    assert "-20.0%p" in text or "+20.0%p" in text


# ---------- detect_big_changes ----------

def test_detect_big_changes_basic():
    new = {"A": 0.4, "B": 0.3, "CASH": 0.3}
    prev = {"A": 0.2, "B": 0.3, "CASH": 0.5}
    out = telegram.detect_big_changes(new, prev, threshold=0.10)
    assert set(out) == {"A", "CASH"}
    assert out["A"] == pytest.approx(0.2)
    assert out["CASH"] == pytest.approx(-0.2)


def test_detect_big_changes_below_threshold_filtered():
    new = {"A": 0.41, "B": 0.59}
    prev = {"A": 0.40, "B": 0.60}
    out = telegram.detect_big_changes(new, prev, threshold=0.10)
    assert out == {}


def test_detect_big_changes_missing_asset():
    """한쪽에만 있는 자산도 변화로 잡힘."""
    new = {"A": 0.5, "B": 0.5}
    prev = {"A": 0.0, "B": 1.0}
    out = telegram.detect_big_changes(new, prev, threshold=0.10)
    assert "A" in out and out["A"] == pytest.approx(0.5)
