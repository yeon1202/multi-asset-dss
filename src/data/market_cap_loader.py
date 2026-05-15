"""
시가총액·발행주식수 로더 — FinanceDataReader 사용.

Phase 2 에서 PER·PBR 계산에 필요:
  PER = 시가총액 / 당기순이익
  PBR = 시가총액 / 자본총계

데이터 출처: `FinanceDataReader.StockListing('KRX')`
  - KOSPI + KOSDAQ 전 종목의 **현재** 스냅샷
  - 컬럼: Code, Name, Marcap(시총, 원), Stocks(상장주식수), Close(종가) 등
  - 무료, API 키 불필요

⚠️ Note (이전 버전):
  - pykrx 1.2.x 가 KRX 엔드포인트 변경으로 작동 불가 (2026-05 기준).
  - FDR 로 전환 후 안정 작동 확인.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd
from loguru import logger

from src.data import cache
from src.utils.config_loader import load_thresholds

# FDR StockListing 원본 컬럼 → 우리 표준 컬럼명
COLUMN_MAP = {
    "Marcap": "market_cap",
    "Stocks": "shares_outstanding",
    "Close": "close",
    "Name": "name",
}


def load_market_cap_snapshot(use_cache: bool = True) -> pd.DataFrame:
    """
    KRX 전 종목(KOSPI + KOSDAQ) 시가총액·발행주식수 현재 스냅샷.

    Returns
    -------
    pd.DataFrame
        index = 종목코드(6자리),
        columns = [market_cap, shares_outstanding, close, name]
    """
    cache_key = "marketcap_krx_snapshot"
    ttl = load_thresholds()["cache"]["price_ttl_days"]

    if use_cache and cache.is_fresh(cache_key, ttl):
        df = cache.load(cache_key)
        if df is not None and not df.empty:
            df.index = df.index.astype(str).str.zfill(6)
            logger.info(f"[캐시] KRX 시가총액 스냅샷 ({len(df)} 종목)")
            return df

    import FinanceDataReader as fdr  # lazy import

    logger.info("[API] FDR KRX 시가총액 스냅샷 다운로드")
    raw = fdr.StockListing("KRX")
    if raw is None or raw.empty:
        raise ValueError("FDR StockListing('KRX') 결과가 비어 있습니다.")

    # 종목코드를 인덱스로
    raw = raw.set_index("Code")
    raw.index = raw.index.astype(str).str.zfill(6)

    # 우리가 쓰는 컬럼만 추려서 이름 표준화
    keep = [c for c in COLUMN_MAP if c in raw.columns]
    df = raw[keep].rename(columns=COLUMN_MAP)

    # 숫자형 강제 변환 (혹시 문자열로 와도 처리)
    for col in ("market_cap", "shares_outstanding", "close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    cache.save(cache_key, df)
    return df


def load_market_caps(
    codes: Iterable[str],
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    지정한 종목만 추려서 시가총액·상장주식수 반환.

    Parameters
    ----------
    codes : iterable of str
        6자리 종목코드.

    Returns
    -------
    pd.DataFrame
        index = code, columns = [market_cap, shares_outstanding, close, name]
        조회 누락 종목은 NaN 행으로 포함.
    """
    snapshot = load_market_cap_snapshot(use_cache=use_cache)
    code_list = [c.zfill(6) for c in codes]

    cols = [c for c in ("market_cap", "shares_outstanding", "close", "name")
            if c in snapshot.columns]
    out = snapshot.loc[snapshot.index.intersection(code_list), cols].copy()

    # 누락된 종목은 NaN 행으로 추가 (PROJECT_SPEC.md §7.3)
    missing = set(code_list) - set(out.index)
    for code in missing:
        out.loc[code] = [float("nan")] * len(cols)
        logger.warning(f"시가총액 조회 누락: {code}")

    return out
