"""
Phase 5 CLI — 백테스팅.

기본: 모멘텀 상위-N 전략을 KODEX 200 buy-and-hold 와 비교.

사용법:
    python -m src.scripts.run_backtest
    python -m src.scripts.run_backtest --start 2020-01-01
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

from src.backtest.costs import CostConfig
from src.backtest.engine import run_backtest
from src.backtest.strategies import buy_and_hold, equal_weight, momentum_top_n
from src.data.price_loader import get_close_matrix, load_universe_prices
from src.utils.config_loader import PROJECT_ROOT, load_backtest_config


def main() -> int:
    parser = argparse.ArgumentParser(description="백테스트 실행")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_backtest_config()
    start_date = pd.to_datetime(args.start or cfg["start_date"])

    logger.info(f"가격 데이터 로드 (시작 {start_date.date()})...")
    # 백테스트 기간이 길면 lookback_days 를 충분히 크게 잡아야 함
    days_needed = (pd.Timestamp.today() - start_date).days + 200
    prices_dict = load_universe_prices(lookback_days=days_needed)
    closes = get_close_matrix(prices_dict)
    closes = closes.loc[closes.index >= start_date]
    if closes.empty:
        logger.error(f"가격 데이터가 {start_date.date()} 이후로 비어있음")
        return 2
    logger.info(f"가격 데이터: {len(closes)} 영업일, {len(closes.columns)} 자산")

    costs = CostConfig(
        commission_rate=cfg["costs"]["commission_rate"],
        slippage_rate=cfg["costs"]["slippage_rate"],
        tax_rate_sell=cfg["costs"]["tax_rate_sell"],
    )

    # 1) 모멘텀 전략
    mom_cfg = cfg["momentum_strategy"]
    mom_strategy = momentum_top_n(
        lookback_days=mom_cfg["lookback_days"],
        top_n=mom_cfg["top_n"],
        cash_when_no_momentum=mom_cfg["cash_when_no_momentum"],
    )
    logger.info("모멘텀 전략 백테스트...")
    result_mom = run_backtest(
        prices=closes, strategy_fn=mom_strategy,
        rebalance_freq=cfg["rebalance_freq"],
        initial_capital=cfg["initial_capital"],
        cost_config=costs, risk_free_rate=cfg["risk_free_rate"],
    )

    # 2) 벤치마크 — KODEX 200 buy-and-hold
    bench_code = cfg["benchmark_code"]
    logger.info(f"벤치마크 백테스트 ({bench_code} buy-and-hold)...")
    result_bench = run_backtest(
        prices=closes, strategy_fn=buy_and_hold(bench_code),
        rebalance_freq=cfg["rebalance_freq"],
        initial_capital=cfg["initial_capital"],
        cost_config=costs, risk_free_rate=cfg["risk_free_rate"],
    )

    # 3) 동일가중 비교
    logger.info("동일가중 비교 백테스트...")
    result_eq = run_backtest(
        prices=closes, strategy_fn=equal_weight(),
        rebalance_freq=cfg["rebalance_freq"],
        initial_capital=cfg["initial_capital"],
        cost_config=costs, risk_free_rate=cfg["risk_free_rate"],
    )

    # 출력
    summary_rows = []
    for name, r in [("모멘텀 Top-N", result_mom),
                     (f"벤치마크 ({bench_code})", result_bench),
                     ("동일가중", result_eq)]:
        m = r.metrics
        summary_rows.append({
            "전략": name,
            "누적수익률": f"{m.total_return*100:+.1f}%",
            "CAGR": f"{m.cagr*100:+.2f}%",
            "변동성": f"{m.volatility*100:.1f}%",
            "샤프": f"{m.sharpe:.2f}",
            "MDD": f"{m.max_drawdown*100:.1f}%",
            "월간 승률": f"{m.win_rate_monthly*100:.1f}%",
            "Calmar": f"{m.calmar:.2f}",
            "거래비용": f"{r.transaction_costs_total:,.0f}원",
        })
    summary = pd.DataFrame(summary_rows)
    logger.info("=== 백테스트 결과 ===")
    print(summary.to_string(index=False))

    # 저장
    today = date.today()
    out_dir = PROJECT_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # equity curve 저장
    equity_df = pd.DataFrame({
        "Momentum": result_mom.equity,
        "Benchmark": result_bench.equity,
        "EqualWeight": result_eq.equity,
    })
    equity_path = args.out or (out_dir / f"backtest_equity_{today}.csv")
    equity_df.to_csv(equity_path, encoding="utf-8-sig")
    logger.info(f"equity curve 저장: {equity_path}")

    summary_path = out_dir / f"backtest_summary_{today}.csv"
    summary.to_csv(summary_path, encoding="utf-8-sig", index=False)
    logger.info(f"성과 요약 저장: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
