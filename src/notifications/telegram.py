"""
Telegram 알림 — Phase 7.

봇 생성:
  1. 텔레그램에서 @BotFather 검색 → /newbot → 봇 이름 + 사용자명 지정
  2. 받은 토큰을 .env 에 TELEGRAM_BOT_TOKEN 으로 저장
  3. 본인이 만든 봇과 대화창 한 번 열고 아무 메시지 전송
  4. https://api.telegram.org/bot<TOKEN>/getUpdates 에서 chat.id 확인
  5. .env 에 TELEGRAM_CHAT_ID 저장
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import requests
from dotenv import load_dotenv
from loguru import logger

from src.utils.config_loader import PROJECT_ROOT, load_notify_config

load_dotenv(PROJECT_ROOT / ".env")


class TelegramKeysMissingError(RuntimeError):
    """TELEGRAM_BOT_TOKEN 또는 TELEGRAM_CHAT_ID 누락."""


def _get_credentials() -> tuple[str, str]:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise TelegramKeysMissingError(
            "TELEGRAM_BOT_TOKEN 또는 TELEGRAM_CHAT_ID 가 설정되지 않았습니다.\n"
            "src/notifications/telegram.py 의 docstring 참고."
        )
    return token, chat_id


def _split_message(text: str, max_len: int) -> list[str]:
    """텔레그램 한도(4096자) 안전하게 분할. 줄 단위로 자름."""
    if len(text) <= max_len:
        return [text]
    parts: list[str] = []
    buf: list[str] = []
    cur_len = 0
    for line in text.splitlines(keepends=True):
        if cur_len + len(line) > max_len and buf:
            parts.append("".join(buf))
            buf = [line]
            cur_len = len(line)
        else:
            buf.append(line)
            cur_len += len(line)
    if buf:
        parts.append("".join(buf))
    return parts


def send_message(
    text: str,
    parse_mode: str = "Markdown",
    token: str | None = None,
    chat_id: str | None = None,
    session: requests.Session | None = None,
) -> list[dict]:
    """
    텔레그램으로 메시지 전송. 길면 분할.

    Parameters
    ----------
    text : 본문 (마크다운 권장)
    parse_mode : "Markdown" | "MarkdownV2" | "HTML" | None
    token, chat_id : 직접 주입 시 환경변수 무시 (테스트용)
    session : requests.Session (테스트 mock 용)

    Returns
    -------
    list of API 응답 dict.
    """
    cfg = load_notify_config()["telegram"]
    if token is None or chat_id is None:
        env_token, env_chat = _get_credentials()
        token = token or env_token
        chat_id = chat_id or env_chat
    s = session or requests.Session()
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    parts = _split_message(text, cfg["max_message_length"])
    responses: list[dict] = []
    for i, part in enumerate(parts):
        payload = {
            "chat_id": chat_id,
            "text": part,
            "disable_web_page_preview": True,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        resp = s.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram API 오류: {data}")
        responses.append(data)
        logger.info(f"Telegram 전송 완료 ({i + 1}/{len(parts)})")
    return responses


def send_report_summary(
    as_of: str,
    regime_label: str,
    regime_score: float,
    allocation: dict[str, float],
    asset_names: dict[str, str] | None = None,
    expected_return: float | None = None,
    sharpe: float | None = None,
    big_changes: dict[str, float] | None = None,
    **send_kwargs,
) -> list[dict]:
    """일일 리포트의 핵심만 추려서 전송."""
    emoji = {"risk_on": "🟢", "risk_off": "🔴", "neutral": "⚪"}.get(regime_label, "❓")
    label_kr = {"risk_on": "RISK-ON", "risk_off": "RISK-OFF",
                "neutral": "NEUTRAL"}.get(regime_label, regime_label)
    names = asset_names or {}

    lines = [
        f"*📊 일일 포트폴리오 — {as_of}*",
        "",
        f"🌐 *시장 국면*: {emoji} {label_kr}  (점수 `{regime_score:+.2f}`)",
        "",
        "*💼 추천 비중*",
    ]
    # 비중 큰 순
    sorted_alloc = sorted(allocation.items(), key=lambda x: -x[1])
    for code, w in sorted_alloc:
        if w < 0.005:
            continue
        if code == "CASH":
            label = "💵 현금"
        else:
            label = names.get(code, code)
        lines.append(f"  • {label}: `{w * 100:.1f}%`")

    if expected_return is not None:
        lines.append("")
        lines.append(f"📈 기대수익률: `{expected_return * 100:+.2f}%` (연)")
    if sharpe is not None:
        lines.append(f"⚡ 샤프: `{sharpe:.2f}`")

    if big_changes:
        lines.append("")
        lines.append("*🔄 큰 변화*")
        for code, diff in big_changes.items():
            label = "💵 현금" if code == "CASH" else names.get(code, code)
            lines.append(f"  • {label}: `{diff * 100:+.1f}%p`")

    lines.append("")
    lines.append("_⚠️ 정보 제공 목적 — 투자 결정은 본인 책임_")

    return send_message("\n".join(lines), **send_kwargs)


def detect_big_changes(
    new_alloc: dict[str, float],
    prev_alloc: dict[str, float],
    threshold: float = 0.10,
) -> dict[str, float]:
    """비중 변화가 threshold 이상인 자산만 반환."""
    keys = set(new_alloc) | set(prev_alloc)
    out: dict[str, float] = {}
    for k in keys:
        diff = new_alloc.get(k, 0.0) - prev_alloc.get(k, 0.0)
        if abs(diff) >= threshold:
            out[k] = diff
    return out
