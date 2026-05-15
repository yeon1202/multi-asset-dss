"""
가격 데이터 수집 모듈 (Phase 1).

FinanceDataReader를 통해 일봉 OHLCV 데이터를 받아옵니다.
캐싱은 src.data.cache 가 담당.

OHLCV = Open(시가), High(고가), Low(저가), Close(종가), Volume(거래량)
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

import pandas as pd
from loguru import logger  # print() 대신 사용. 시간·레벨·파일 위치 자동 기록

from src.data import cache
from src.utils.config_loader import load_thresholds, load_universe


def _today() -> date:
    """오늘 날짜 — 테스트에서 monkeypatch 하기 쉽게 함수로 분리."""
    return date.today()


def load_price(
    code: str,
    lookback_days: int | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    단일 종목 일봉 데이터를 DataFrame으로 반환.

    Parameters
    ----------
    code : str
        종목코드 (예: "069500" KODEX 200) 또는 티커 (예: "AAPL").
    lookback_days : int, optional
        조회 일수. 기본은 universe.yaml의 default_lookback_days.
    use_cache : bool
        True면 TTL 내 캐시 우선 사용.

    Returns
    -------
    pd.DataFrame
        index = Date, columns = [Open, High, Low, Close, Volume, Change].
    """
    if lookback_days is None:
        lookback_days = load_universe().get("default_lookback_days", 365)

    cache_key = f"price_{code}_{lookback_days}d"
    ttl = load_thresholds()["cache"]["price_ttl_days"]

    if use_cache and cache.is_fresh(cache_key, ttl):
        df = cache.load(cache_key)
        if df is not None and not df.empty:
            logger.info(f"[캐시] {code} ({len(df)} rows)")
            return df

    # FinanceDataReader는 무거우므로 함수 내부에서 lazy import
    import FinanceDataReader as fdr

    end = _today()
    start = end - timedelta(days=lookback_days)

    logger.info(f"[API] {code} {start} ~ {end} 다운로드 중")
    df = fdr.DataReader(code, start, end)

    if df is None or df.empty:
        raise ValueError(f"{code} 의 가격 데이터가 비어 있습니다.")

    cache.save(cache_key, df)
    return df


def load_universe_prices(
    lookback_days: int | None = None,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    universe.yaml의 모든 자산을 한 번에 로드.

    Returns
    -------
    dict[str, pd.DataFrame]
        {종목코드: 가격 DataFrame}
    """
    universe = load_universe()
    out: dict[str, pd.DataFrame] = {}
    for asset in universe["assets"]:
        code = asset["code"]
        try:
            out[code] = load_price(code, lookback_days, use_cache)
        except Exception as e:  # noqa: BLE001
            # 하나가 실패해도 나머지는 진행 — 대시보드에서 누락 표시
            logger.error(f"{code} 로드 실패: {e}")
    return out


def get_close_matrix(
    prices: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    여러 종목의 종가만 모은 wide-format DataFrame.

    columns = 종목코드, index = Date, values = Close

    이 형태가 차트·상관관계·정규화에 편함.
    """
    if prices is None:
        prices = load_universe_prices()
    # concat(axis=1): 컬럼 방향으로 합치기
    closes = pd.concat(
        {code: df["Close"] for code, df in prices.items()},
        axis=1,
    )
    closes.index = pd.to_datetime(closes.index)
    return closes.sort_index()
