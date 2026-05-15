"""
시가총액·발행주식수 로더 — pykrx 사용.

Phase 2 에서 PER·PBR 계산에 필요:
  PER = 시가총액 / 당기순이익
  PBR = 시가총액 / 자본총계

pykrx 는 한국거래소(KRX) 공시 데이터를 받아옴 (무료, API 키 불필요).
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

import pandas as pd
from loguru import logger

from src.data import cache
from src.utils.config_loader import load_thresholds


def _latest_business_day() -> str:
    """가장 최근 영업일 추정 — pykrx 가 인식하는 YYYYMMDD 형태."""
    d = date.today()
    # 주말이면 금요일로
    while d.weekday() >= 5:  # 5=토, 6=일
        d -= timedelta(days=1)
    return d.strftime("%Y%m%d")


def load_market_cap_snapshot(
    yyyymmdd: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    특정 일자의 KOSPI 전 종목 시가총액 스냅샷.

    Parameters
    ----------
    yyyymmdd : str, optional
        "20250513" 같은 날짜. 기본은 최근 영업일.

    Returns
    -------
    pd.DataFrame
        index = 종목코드(6자리),
        columns = [시가총액, 거래량, 거래대금, 상장주식수].
    """
    yyyymmdd = yyyymmdd or _latest_business_day()
    cache_key = f"marketcap_{yyyymmdd}"
    ttl = load_thresholds()["cache"]["price_ttl_days"]

    if use_cache and cache.is_fresh(cache_key, ttl):
        df = cache.load(cache_key)
        if df is not None and not df.empty:
            logger.info(f"[캐시] 시가총액 {yyyymmdd} ({len(df)} 종목)")
            # 인덱스는 종목코드 — read_csv 시 문자열로 강제
            df.index = df.index.astype(str).str.zfill(6)
            return df

    from pykrx import stock  # lazy import

    logger.info(f"[API] 시가총액 스냅샷 {yyyymmdd}")
    df = stock.get_market_cap_by_ticker(yyyymmdd)
    if df is None or df.empty:
        raise ValueError(f"pykrx 시가총액 결과가 비어 있습니다 ({yyyymmdd})")

    df.index = df.index.astype(str).str.zfill(6)
    cache.save(cache_key, df)
    return df


def load_market_caps(
    codes: Iterable[str],
    yyyymmdd: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    지정한 종목만 추려서 시가총액·상장주식수 반환.

    Returns
    -------
    pd.DataFrame
        index = code, columns = [market_cap, shares_outstanding]
    """
    snapshot = load_market_cap_snapshot(yyyymmdd, use_cache)
    # 컬럼명이 한글일 수도 있음 → 영문 표준화
    rename = {
        "시가총액": "market_cap",
        "상장주식수": "shares_outstanding",
        "종가": "close",
    }
    snapshot = snapshot.rename(columns=rename)
    cols = [c for c in ("market_cap", "shares_outstanding", "close") if c in snapshot.columns]
    out = snapshot.loc[snapshot.index.intersection(list(codes)), cols].copy()
    # 누락된 종목은 NaN 행으로 추가
    missing = set(codes) - set(out.index)
    for code in missing:
        out.loc[code] = [float("nan")] * len(cols)
        logger.warning(f"시가총액 조회 누락: {code}")
    return out
