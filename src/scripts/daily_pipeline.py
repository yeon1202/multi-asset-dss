"""
Phase 7 일일 파이프라인 — 모든 단계를 순차 실행 + 텔레그램 알림.

cron / Task Scheduler / GitHub Actions 에서 호출.

사용법:
    python -m src.scripts.daily_pipeline
    python -m src.scripts.daily_pipeline --skip-notify
    python -m src.scripts.daily_pipeline --no-cache
"""
from __future__ import annotations

import argparse
import sys
from datetime import date

from loguru import logger

from src.scripts.run_daily_report import main as run_report_main
from src.scripts.run_stock_portfolio import main as run_stock_main
from src.utils.config_loader import PROJECT_ROOT, load_notify_config, load_universe


def main() -> int:
    parser = argparse.ArgumentParser(description="일일 파이프라인 (Phase 7)")
    parser.add_argument("--skip-notify", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args, unknown = parser.parse_known_args()

    # 1a) ETF 일일 리포트 생성 (Phase 1 → 3 → 4)
    logger.info("=== 단계 1a: ETF 포트폴리오 리포트 (Phase 4) ===")
    cli_args = []
    if args.no_cache:
        cli_args.append("--no-cache")
    sys.argv = ["run_daily_report"] + cli_args
    report_code = run_report_main()
    if report_code != 0:
        logger.error(f"ETF 리포트 생성 실패 (exit {report_code})")
        return report_code

    # 1b) 한국 주식 포트폴리오 리포트 (Phase 2 → 3 → 8)
    logger.info("=== 단계 1b: 한국 주식 포트폴리오 (Phase 8) ===")
    sys.argv = ["run_stock_portfolio"] + cli_args
    try:
        stock_code = run_stock_main()
        if stock_code != 0:
            logger.warning(f"주식 포트폴리오 생성 비정상 종료 (exit {stock_code}) — ETF 만으로 계속")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"주식 포트폴리오 실패 — ETF 만으로 계속: {e}")

    # 2) 텔레그램 알림
    if args.skip_notify:
        logger.info("--skip-notify 지정 — 알림 생략")
        return 0

    notify_cfg = load_notify_config()
    if not notify_cfg["telegram"].get("enabled", True):
        logger.info("notify.yaml 에서 telegram disabled")
        return 0

    logger.info("=== 단계 2: 텔레그램 알림 ===")
    try:
        from src.notifications.telegram import (
            TelegramKeysMissingError,
            detect_big_changes,
            send_report_summary,
        )

        # 최신 리포트 위치 — 오늘 날짜의 .md
        today = date.today()
        report_path = PROJECT_ROOT / "reports" / f"{today}.md"
        if not report_path.exists():
            logger.warning(f"리포트 파일 없음: {report_path}")
            return 1

        # 리포트에서 핵심 추출이 복잡하니, 파이프라인을 다시 안 돌리고
        # 단순히 리포트 내용을 그대로 전송 (max length 자동 분할)
        report_text = report_path.read_text(encoding="utf-8")
        from src.notifications.telegram import send_message
        send_message(report_text, parse_mode=None)
        logger.info("✅ 텔레그램 전송 완료")
    except TelegramKeysMissingError as e:
        logger.warning(str(e))
        return 0  # 키 없으면 그냥 알림만 skip
    except Exception as e:  # noqa: BLE001
        logger.error(f"알림 실패: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
