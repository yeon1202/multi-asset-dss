"""
Phase 2 CLI — 펀더멘털 스코어링 일괄 실행.

사용법:
    python -m src.scripts.run_fundamental_scoring
    python -m src.scripts.run_fundamental_scoring --year 2024 --top 10
    python -m src.scripts.run_fundamental_scoring --no-cache

`reports/fundamental_YYYY-MM-DD.csv` 로 결과 저장.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

from src.data.dart_loader import DartKeyMissingError, load_universe_financials
from src.data.market_cap_loader import load_market_caps
from src.indicators.fundamental import compute_ratios_table
from src.scoring.fundamental_score import fundamental_score, top_n
from src.utils.config_loader import PROJECT_ROOT, load_fundamental_config


def main() -> int:
    cfg = load_fundamental_config()
    parser = argparse.ArgumentParser(description="펀더멘털 스코어링 실행")
    parser.add_argument("--year", type=int, default=cfg["target_year"])
    parser.add_argument("--top", type=int, default=cfg.get("top_n", 10))
    parser.add_argument("--no-cache", action="store_true", help="캐시 무시")
    parser.add_argument("--out", type=Path, default=None, help="결과 CSV 경로")
    args = parser.parse_args()

    codes = [a["code"] for a in cfg["universe"]]
    name_map = {a["code"]: a["name"] for a in cfg["universe"]}
    sector_map = {a["code"]: a["sector"] for a in cfg["universe"]}

    logger.info(f"유니버스 {len(codes)} 종목 · 회계연도 {args.year}")

    # 1) 재무제표 (DART)
    try:
        fin = load_universe_financials(
            codes,
            year=args.year,
            report_type=cfg["report_type"],
            use_cache=not args.no_cache,
        )
    except DartKeyMissingError as e:
        logger.error(str(e))
        return 2  # exit code 2 = config error

    # 2) 시가총액 (pykrx)
    mcap = load_market_caps(codes, use_cache=not args.no_cache)

    # 3) 비율 계산
    ratios = compute_ratios_table(fin, mcap)

    # 4) 점수화
    scored = fundamental_score(
        ratios,
        weights=cfg["score_weights"],
        sanity_bounds=cfg.get("sanity_bounds"),
    )

    # 5) 보기 좋게 이름·섹터 추가
    scored.insert(0, "name", scored.index.map(name_map))
    scored.insert(1, "sector", scored.index.map(sector_map))

    # 6) 상위 N 출력
    leaders = top_n(scored, n=args.top)
    logger.info(f"상위 {args.top} 종목:")
    print(leaders.to_string())

    # 7) 저장
    out_path: Path = args.out or (PROJECT_ROOT / "reports" / f"fundamental_{date.today()}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_path, encoding="utf-8-sig")
    logger.info(f"결과 저장: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
