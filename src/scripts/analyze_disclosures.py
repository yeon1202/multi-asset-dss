"""
Phase 6 CLI — DART 최근 공시 분석.

사용법:
    python -m src.scripts.analyze_disclosures
    python -m src.scripts.analyze_disclosures --code 005930 --days 30
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

import pandas as pd
from loguru import logger

from src.data.dart_loader import DartKeyMissingError, get_api_key as get_dart_key
from src.llm.disclosure_analyzer import (
    AnthropicKeyMissingError,
    analyze_disclosure,
    estimate_cost_usd,
)
from src.scoring.disclosure_score import disclosure_score_table
from src.utils.config_loader import PROJECT_ROOT, load_fundamental_config, load_llm_config


def main() -> int:
    parser = argparse.ArgumentParser(description="DART 공시 LLM 분석")
    parser.add_argument("--code", type=str, default=None, help="단일 종목코드")
    parser.add_argument("--days", type=int, default=30, help="최근 N일")
    parser.add_argument("--max", type=int, default=10, help="최대 분석 건수")
    args = parser.parse_args()

    # API 키 체크
    try:
        get_dart_key()
    except DartKeyMissingError as e:
        logger.error(str(e))
        return 2

    cfg = load_llm_config()

    # 종목 리스트
    if args.code:
        codes = [args.code]
    else:
        fund_cfg = load_fundamental_config()
        codes = [a["code"] for a in fund_cfg["universe"][:5]]  # 상위 5개

    # DART에서 최근 공시 목록 조회
    from opendartreader import OpenDartReader
    dart = OpenDartReader(get_dart_key())

    end_date = date.today()
    start_date = end_date - timedelta(days=args.days)

    analyses_by_code: dict[str, list[dict]] = {}
    total_cost = 0.0
    count = 0

    for code in codes:
        logger.info(f"=== {code} 공시 조회 ===")
        try:
            listing = dart.list(code, start=start_date.isoformat(), end=end_date.isoformat())
        except Exception as e:  # noqa: BLE001
            logger.error(f"{code} 공시 목록 조회 실패: {e}")
            continue

        if listing is None or listing.empty:
            logger.info(f"{code}: 공시 없음")
            analyses_by_code[code] = []
            continue

        # 최대 max 건만 분석
        listing = listing.head(args.max)
        results = []
        for _, row in listing.iterrows():
            if count >= args.max * len(codes):
                break
            try:
                content = dart.document(row["rcept_no"])
                content_text = str(content)[:10000] if content else ""
                if not content_text:
                    continue
                try:
                    result = analyze_disclosure(
                        corp_name=row["corp_name"],
                        stock_code=code,
                        report_name=row["report_nm"],
                        report_date=row["rcept_dt"],
                        content=content_text,
                    )
                except AnthropicKeyMissingError as e:
                    logger.error(str(e))
                    return 2

                cost = estimate_cost_usd(
                    result.input_tokens, result.output_tokens, result.model
                )
                total_cost += cost
                count += 1

                logger.info(
                    f"  {row['report_nm'][:30]:30s}  "
                    f"sentiment={result.sentiment_score:+.2f} "
                    f"({result.sentiment_label})  "
                    f"~${cost:.4f}"
                )
                results.append({
                    "report_name": row["report_nm"],
                    "report_date": row["rcept_dt"],
                    "summary": result.summary,
                    "sentiment_score": result.sentiment_score,
                    "sentiment_label": result.sentiment_label,
                    "keywords": result.keywords,
                    "reasoning": result.reasoning,
                })
            except Exception as e:  # noqa: BLE001
                logger.error(f"  분석 실패 {row['rcept_no']}: {e}")
        analyses_by_code[code] = results

    # 집계
    scores = disclosure_score_table(analyses_by_code)
    logger.info("=== 종목별 정성 점수 ===")
    print(scores.to_string())
    logger.info(f"총 비용 추정: ${total_cost:.4f}")
    if total_cost > cfg["cost_monitor"]["alert_threshold_usd"]:
        logger.warning(
            f"⚠️ 경고 임계({cfg['cost_monitor']['alert_threshold_usd']} USD) 초과!"
        )

    # 저장
    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today()
    scores_path = out_dir / f"disclosure_scores_{today}.csv"
    scores.to_csv(scores_path, encoding="utf-8-sig")
    logger.info(f"결과 저장: {scores_path}")

    # 상세 결과 (JSON Lines)
    import json
    detail_path = out_dir / f"disclosure_analyses_{today}.jsonl"
    with detail_path.open("w", encoding="utf-8") as f:
        for code, results in analyses_by_code.items():
            for r in results:
                f.write(json.dumps({"code": code, **r}, ensure_ascii=False) + "\n")
    logger.info(f"상세 결과 저장: {detail_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
