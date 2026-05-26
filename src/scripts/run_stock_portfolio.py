"""
Phase 8 CLI — 한국 주식 전용 포트폴리오 추천 생성.

파이프라인:
  1. Phase 2 펀더멘털 점수 (DART + FDR 시총) — 30종목
  2. 종목별 일봉 가격 (FDR)
  3. Phase 3 레짐 (ECOS + FRED)
  4. 통합 점수 = 기술 + 펀더멘털 + 레짐 (섹터별 가중)
  5. 마코위츠 max-Sharpe (자산당 10%, 현금 ≥ 10%)
  6. Half-Kelly 사이징
  7. Markdown 리포트 + 자동 해설

사용법:
    python -m src.scripts.run_stock_portfolio
    python -m src.scripts.run_stock_portfolio --no-cache --year 2024
"""
from __future__ import annotations

import argparse
import sys
from datetime import date

import pandas as pd
from loguru import logger

from src.data.dart_loader import DartKeyMissingError, load_universe_financials
from src.data.macro_loader import (
    EcosKeyMissingError,
    FredKeyMissingError,
    load_all_macro,
)
from src.data.market_cap_loader import load_market_caps
from src.data.price_loader import load_price
from src.indicators.fundamental import compute_ratios_table
from src.portfolio.kelly import apply_kelly_sizing
from src.portfolio.optimizer import equal_weight_fallback, optimize_max_sharpe
from src.regime.detector import classify_regime
from src.report.generator import ReportContext
from src.report.narrative import narrate_all
from src.scoring.fundamental_score import fundamental_score
from src.scoring.stock_portfolio_score import stock_composite_score
from src.utils.config_loader import (
    PROJECT_ROOT,
    load_fundamental_config,
    load_macro_config,
    load_stock_portfolio_config,
)


def _load_stock_prices(codes: list[str], lookback_days: int = 365) -> pd.DataFrame:
    """종목 코드별 종가 시계열을 wide DataFrame 으로."""
    closes: dict[str, pd.Series] = {}
    for code in codes:
        try:
            df = load_price(code, lookback_days=lookback_days)
            closes[code] = df["Close"]
        except Exception as e:  # noqa: BLE001
            logger.error(f"가격 로드 실패 {code}: {e}")
    if not closes:
        raise RuntimeError("종목 가격을 하나도 로드하지 못함")
    out = pd.concat(closes, axis=1)
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def main() -> int:
    parser = argparse.ArgumentParser(description="한국 주식 포트폴리오 추천")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--year", type=int, default=None, help="펀더멘털 회계연도")
    args = parser.parse_args()

    fund_cfg = load_fundamental_config()
    macro_cfg = load_macro_config()
    stock_cfg = load_stock_portfolio_config()

    codes = [a["code"] for a in fund_cfg["universe"]]
    name_map = {a["code"]: a["name"] for a in fund_cfg["universe"]}
    sector_map = {a["code"]: a["sector"] for a in fund_cfg["universe"]}
    year = args.year or fund_cfg["target_year"]

    # 1) 펀더멘털 점수 (Phase 2 파이프라인 재사용)
    logger.info(f"펀더멘털 점수 계산 ({len(codes)}종목, {year}년)...")
    try:
        fin = load_universe_financials(
            codes, year=year, report_type=fund_cfg["report_type"],
            use_cache=not args.no_cache,
        )
    except DartKeyMissingError as e:
        logger.error(str(e))
        return 2
    mcap = load_market_caps(codes, use_cache=not args.no_cache)
    ratios = compute_ratios_table(fin, mcap)
    fund_scored = fundamental_score(
        ratios, weights=fund_cfg["score_weights"],
        sanity_bounds=fund_cfg.get("sanity_bounds"),
    )
    fund_composite = fund_scored["composite_score"]

    # 2) 종목 가격 — 1년치
    logger.info("종목 가격 다운로드...")
    prices = _load_stock_prices(codes)
    # 종목 중 가격 못 받은 것 제외
    valid_codes = [c for c in codes if c in prices.columns]
    logger.info(f"가격 확보: {len(valid_codes)}/{len(codes)}종목")

    # 3) 레짐 (Phase 3 파이프라인)
    logger.info("레짐 판정...")
    try:
        macro = load_all_macro(use_cache=not args.no_cache)
    except (EcosKeyMissingError, FredKeyMissingError) as e:
        logger.error(str(e))
        return 2
    latest = macro.iloc[-1]
    feature_values = {n: latest.get(n, float("nan"))
                      for n in macro_cfg["regime"]["features"]}
    regime_res = classify_regime(feature_values, macro_cfg["regime"])
    logger.info(f"레짐: {regime_res.label}  (score={regime_res.score:+.3f})")

    # 4) 통합 점수
    logger.info("종목 통합 점수 산출...")
    scores = stock_composite_score(
        prices=prices,
        fundamental_scores=fund_composite,
        regime_score=regime_res.score,
        sector_map=sector_map,
        sector_preferences=stock_cfg["sector_preferences"],
        default_preference=stock_cfg["default_preference"],
        weights=stock_cfg["score_weights"],
        technical_cfg=stock_cfg["technical_score"],
    )
    logger.info(f"\n{scores.round(1).sort_values('composite', ascending=False).head(10)}")

    # 5) 마코위츠 최적화
    logger.info("마코위츠 최적화...")
    constraints = stock_cfg["constraints"]
    composite_for_opt = scores["composite"].dropna()
    prices_for_opt = prices[composite_for_opt.index]
    try:
        opt = optimize_max_sharpe(
            prices=prices_for_opt,
            composite_score=composite_for_opt,
            max_expected_return=stock_cfg["expected_return_mapping"]["max_expected_return"],
            max_weight_per_asset=constraints["max_weight_per_asset"],
            min_weight_per_asset=constraints["min_weight_per_asset"],
            risk_free_rate=constraints["risk_free_rate"],
        )
        risky = opt.weights
        port_ret, port_vol, port_sharpe = opt.expected_return, opt.volatility, opt.sharpe
    except Exception as e:  # noqa: BLE001
        logger.warning(f"최적화 실패 — 동일가중 fallback: {e}")
        risky = equal_weight_fallback(composite_for_opt)
        daily = prices_for_opt.pct_change().dropna()
        port_ret = float((risky * daily.mean()).sum() * 252)
        port_vol = float((daily @ risky.values).std() * (252 ** 0.5))
        port_sharpe = (port_ret - constraints["risk_free_rate"]) / port_vol if port_vol > 0 else 0.0

    # 6) Half-Kelly
    logger.info("Half-Kelly 사이징...")
    kelly_cfg = stock_cfg["kelly"]
    final_alloc = apply_kelly_sizing(
        risky_weights=risky,
        expected_return=port_ret, volatility=port_vol,
        risk_free_rate=constraints["risk_free_rate"],
        kelly_fraction_param=kelly_cfg["fraction"],
        max_total_risk_weight=kelly_cfg["max_total_risk_weight"],
        min_cash_weight=constraints["min_cash_weight"],
    )
    cash_pct = final_alloc.get("CASH", 0.0) * 100
    logger.info(f"최종 배분: 위험자산 {100 - cash_pct:.1f}%, 현금 {cash_pct:.1f}%")
    kelly_scale = 1.0 - final_alloc.get("CASH", 0.0)

    # 7) 리포트
    ctx = ReportContext(
        as_of=date.today(),
        regime_label=regime_res.label,
        regime_score=regime_res.score,
        regime_contributions=regime_res.contributions,
        regime_feature_values=regime_res.feature_values,
        scores=scores,
        allocation=final_alloc,
        asset_names=name_map,
        expected_return=port_ret,
        volatility=port_vol,
        sharpe=port_sharpe,
        kelly_fraction=kelly_scale,
        prev_allocation=None,
    )
    today = date.today()
    out_path = PROJECT_ROOT / "reports" / f"stock_portfolio_{today}.md"
    from src.report.generator import render
    out_path.write_text(render(ctx), encoding="utf-8")
    logger.info(f"리포트 저장: {out_path}")

    # 상위 N 콘솔 출력
    leaders = final_alloc.drop("CASH", errors="ignore").sort_values(ascending=False).head(10)
    print("\n=== 상위 10 추천 종목 ===")
    for code, w in leaders.items():
        if w < 0.005:
            break
        name = name_map.get(code, code)
        sector = sector_map.get(code, "-")
        s = scores.loc[code]
        print(f"  {name:14s} ({sector:8s}) {w * 100:5.2f}%  "
              f"composite={s['composite']:5.1f}  "
              f"(T:{s['technical']:.0f} / F:{s.get('fundamental', float('nan')):.0f} / R:{s['regime']:.0f})")
    # 이모지 빼고 — PowerShell cp949 인코딩 호환
    print(f"  [현금]                       {cash_pct:5.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
