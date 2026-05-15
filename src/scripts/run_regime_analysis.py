"""
Phase 3 CLI — 거시 데이터 수집 + 레짐 판정.

사용법:
    python -m src.scripts.run_regime_analysis
    python -m src.scripts.run_regime_analysis --no-cache

결과:
    reports/regime_YYYY-MM-DD.csv  (현재 레짐 + 자산 적합도)
    reports/regime_history_YYYY-MM-DD.csv  (일별 히스토리)
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

from src.data.macro_loader import (
    EcosKeyMissingError,
    FredKeyMissingError,
    load_all_macro,
)
from src.regime.detector import classify_regime, detect_history
from src.scoring.regime_fit import asset_fit_table
from src.utils.config_loader import PROJECT_ROOT, load_macro_config


def main() -> int:
    parser = argparse.ArgumentParser(description="레짐 판정 + 자산 적합도")
    parser.add_argument("--no-cache", action="store_true", help="캐시 무시")
    args = parser.parse_args()

    cfg = load_macro_config()

    try:
        macro = load_all_macro(use_cache=not args.no_cache)
    except (EcosKeyMissingError, FredKeyMissingError) as e:
        logger.error(str(e))
        return 2

    # 가장 최근 시점
    latest = macro.iloc[-1]
    latest_date = latest.name
    values = {name: latest.get(name, float("nan")) for name in cfg["regime"]["features"]}
    result = classify_regime(values, cfg["regime"])

    logger.info(f"=== 현재 레짐 ({latest_date.date()}) ===")
    logger.info(f"종합 점수: {result.score:+.3f}  →  라벨: {result.label.upper()}")
    logger.info("기여도:")
    for feat, contrib in sorted(result.contributions.items(), key=lambda x: -abs(x[1])):
        val = result.feature_values.get(feat)
        score = result.feature_scores.get(feat)
        logger.info(
            f"  {feat:12s} value={val:>8.3f}  score={score:+.2f}  기여={contrib:+.3f}"
        )

    # 자산 적합도
    fit = asset_fit_table(cfg["asset_preference"], result.score)
    fit = fit.sort_values("regime_fit", ascending=False)
    logger.info("=== 자산 레짐 적합도 ===")
    print(fit.to_string())

    # 히스토리
    history = detect_history(macro, cfg["regime"])

    # 저장
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    today = date.today()

    snapshot_rows = [
        {"feature": k, "value": result.feature_values.get(k),
         "score": result.feature_scores.get(k),
         "contribution": result.contributions.get(k)}
        for k in cfg["regime"]["features"]
    ]
    snapshot = pd.DataFrame(snapshot_rows)
    snapshot.loc[len(snapshot)] = ["COMPOSITE", None, None, result.score]
    snapshot.loc[len(snapshot)] = ["LABEL", None, None, result.label]

    snap_path = reports_dir / f"regime_{today}.csv"
    snapshot.to_csv(snap_path, index=False, encoding="utf-8-sig")
    logger.info(f"스냅샷 저장: {snap_path}")

    hist_path = reports_dir / f"regime_history_{today}.csv"
    history.to_csv(hist_path, encoding="utf-8-sig")
    logger.info(f"히스토리 저장: {hist_path}")

    fit_path = reports_dir / f"regime_fit_{today}.csv"
    fit.to_csv(fit_path, encoding="utf-8-sig")
    logger.info(f"자산 적합도 저장: {fit_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
