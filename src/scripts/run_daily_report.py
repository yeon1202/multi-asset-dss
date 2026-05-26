"""
Phase 4 CLI — 일일 포트폴리오 리포트 생성.

전체 파이프라인:
  1. 가격 데이터 로드 (Phase 1)
  2. 거시 데이터 로드 + 레짐 판정 (Phase 3)
  3. 통합 점수 산출 (Phase 4)
  4. 마코위츠 최적화 (Phase 4)
  5. Half-Kelly 사이징 (Phase 4)
  6. Markdown 리포트 저장

사용법:
    python -m src.scripts.run_daily_report
    python -m src.scripts.run_daily_report --no-cache
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
from src.data.price_loader import get_close_matrix, load_universe_prices
from src.portfolio.kelly import apply_kelly_sizing
from src.portfolio.optimizer import (
    equal_weight_fallback,
    optimize_max_sharpe,
)
from src.regime.detector import classify_regime
from src.report.generator import ReportContext, write_report
from src.scoring.composite_score import composite_score
from src.utils.config_loader import (
    PROJECT_ROOT,
    load_macro_config,
    load_portfolio_config,
    load_universe,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="일일 포트폴리오 리포트 생성")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    universe = load_universe()
    macro_cfg = load_macro_config()
    port_cfg = load_portfolio_config()

    # 1) 가격
    logger.info("가격 데이터 로드...")
    prices_dict = load_universe_prices(use_cache=not args.no_cache)
    closes = get_close_matrix(prices_dict)

    # 2) 거시 + 레짐
    logger.info("거시 데이터 로드 + 레짐 판정...")
    try:
        macro = load_all_macro(use_cache=not args.no_cache)
    except (EcosKeyMissingError, FredKeyMissingError) as e:
        logger.error(str(e))
        return 2

    latest_macro = macro.iloc[-1]
    feature_values = {
        n: latest_macro.get(n, float("nan"))
        for n in macro_cfg["regime"]["features"]
    }
    regime_res = classify_regime(feature_values, macro_cfg["regime"])
    logger.info(
        f"레짐: {regime_res.label}  (score={regime_res.score:+.3f})"
    )

    # 3) 통합 점수
    logger.info("통합 점수 산출...")
    scores = composite_score(
        prices=closes,
        regime_score=regime_res.score,
        preferences=macro_cfg["asset_preference"],
        weights=port_cfg["score_weights"],
        technical_cfg=port_cfg["technical_score"],
    )
    logger.info(f"\n{scores.round(1)}")

    # 4) 마코위츠
    logger.info("마코위츠 최적화...")
    constraints = port_cfg["constraints"]
    try:
        opt = optimize_max_sharpe(
            prices=closes,
            composite_score=scores["composite"],
            max_expected_return=port_cfg["expected_return_mapping"]["max_expected_return"],
            max_weight_per_asset=constraints["max_weight_per_asset"],
            min_weight_per_asset=constraints["min_weight_per_asset"],
            risk_free_rate=constraints["risk_free_rate"],
        )
        risky_weights = opt.weights
        port_ret = opt.expected_return
        port_vol = opt.volatility
        port_sharpe = opt.sharpe
        logger.info(
            f"위험자산 포트폴리오: μ={port_ret*100:.2f}%, "
            f"σ={port_vol*100:.2f}%, Sharpe={port_sharpe:.2f}"
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"최적화 실패 — 동일가중 fallback: {e}")
        risky_weights = equal_weight_fallback(scores["composite"])
        # 간이 추정
        daily_ret = closes.pct_change().dropna()
        port_ret = (risky_weights * daily_ret.mean()).sum() * 252
        port_vol = float(((daily_ret @ risky_weights.values).std()) * (252 ** 0.5))
        port_sharpe = (port_ret - constraints["risk_free_rate"]) / port_vol if port_vol > 0 else 0.0

    # 5) Half-Kelly
    logger.info("Half-Kelly 사이징...")
    kelly_cfg = port_cfg["kelly"]
    final_alloc = apply_kelly_sizing(
        risky_weights=risky_weights,
        expected_return=port_ret,
        volatility=port_vol,
        risk_free_rate=constraints["risk_free_rate"],
        kelly_fraction_param=kelly_cfg["fraction"],
        max_total_risk_weight=kelly_cfg["max_total_risk_weight"],
        min_cash_weight=constraints["min_cash_weight"],
    )
    cash_pct = final_alloc.get("CASH", 0.0) * 100
    risky_pct = 100 - cash_pct
    logger.info(f"최종 배분: 위험자산 {risky_pct:.1f}%, 현금 {cash_pct:.1f}%")

    # Kelly 스케일 = (1 - cash) — 결과로부터 역추정
    kelly_scale = 1.0 - final_alloc.get("CASH", 0.0)

    # 6) 직전 리포트 비교
    prev_alloc = _load_previous_allocation()

    # 7) 리포트 작성
    asset_names = {a["code"]: a["name"] for a in universe["assets"]}
    ctx = ReportContext(
        as_of=date.today(),
        regime_label=regime_res.label,
        regime_score=regime_res.score,
        regime_contributions=regime_res.contributions,
        regime_feature_values=regime_res.feature_values,
        scores=scores,
        allocation=final_alloc,
        asset_names=asset_names,
        expected_return=port_ret,
        volatility=port_vol,
        sharpe=port_sharpe,
        kelly_fraction=kelly_scale,
        prev_allocation=prev_alloc,
    )
    out_path = args.out or (PROJECT_ROOT / "reports" / f"{date.today()}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(__import__("src.report.generator", fromlist=["render"]).render(ctx), encoding="utf-8")
    logger.info(f"리포트 저장: {out_path}")
    return 0


def _load_previous_allocation() -> pd.Series | None:
    """가장 최근 reports/YYYY-MM-DD.md 가 아닌 그 이전 리포트의 비중 추출.

    간단 구현: 직전 일자 리포트가 없으면 None 반환.
    Phase 5+ 에서 별도 비중 이력 CSV 를 따로 관리하도록 고도화 권장.
    """
    return None  # MVP: 이력 추적은 Phase 7 자동화에서 보강


if __name__ == "__main__":
    sys.exit(main())
