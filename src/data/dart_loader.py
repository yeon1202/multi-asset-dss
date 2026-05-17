"""
DART (전자공시시스템) 재무제표 로더 — Phase 2.

DART = Data Analysis, Retrieval and Transfer System (금감원 전자공시).
API 키 발급: https://opendart.fss.or.kr  → "오픈API" → "인증키 신청"

PROJECT_SPEC.md §8 보안 원칙: 키는 절대 코드에 하드코딩 X.
→ 프로젝트 루트의 `.env` 파일에 `DART_API_KEY=...` 형태로 저장.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Iterable

import pandas as pd
from dotenv import load_dotenv  # .env 파일을 환경변수로 로드
from loguru import logger

from src.data import cache
from src.utils.config_loader import PROJECT_ROOT, load_thresholds

# `.env` 가 있으면 환경변수에 추가. 없으면 무시 (테스트 환경)
load_dotenv(PROJECT_ROOT / ".env")

# DART finstate 가 반환하는 핵심 계정과목명
# (회사마다 표기가 약간 다를 수 있어 set으로 변형 허용)
ACCOUNT_ALIASES: dict[str, set[str]] = {
    "revenue":      {"매출액", "수익(매출액)", "영업수익"},
    "op_profit":    {"영업이익", "영업이익(손실)"},
    "net_income":   {"당기순이익", "당기순이익(손실)", "당기순손익"},
    "total_assets": {"자산총계"},
    "total_debt":   {"부채총계"},
    "total_equity": {"자본총계"},
}


class DartKeyMissingError(RuntimeError):
    """DART_API_KEY 가 .env 또는 환경변수에 없을 때."""


def get_api_key() -> str:
    """환경변수에서 DART API 키 조회. 없으면 에러."""
    key = os.environ.get("DART_API_KEY", "").strip()
    if not key:
        raise DartKeyMissingError(
            "DART_API_KEY 가 설정되지 않았습니다.\n"
            f"1) https://opendart.fss.or.kr 에서 인증키 발급\n"
            f"2) {PROJECT_ROOT}/.env 파일에 'DART_API_KEY=발급받은키' 추가"
        )
    return key


@lru_cache(maxsize=1)
def _client():
    """OpenDartReader 인스턴스. 첫 호출 시에만 만들고 이후엔 캐시 반환.

    `@lru_cache(maxsize=1)` : functools 표준 데코레이터. 함수 결과를 최대 1개
    저장하고 같은 인자(여기선 인자 없음)로 다시 호출되면 캐시된 결과 반환.
    """
    from opendartreader import OpenDartReader  # 함수 안 lazy import
    return OpenDartReader(get_api_key())


def _normalize_account_name(name: str) -> str | None:
    """DART의 계정명을 표준 키(revenue, op_profit 등)로 매핑."""
    for std, aliases in ACCOUNT_ALIASES.items():
        if name in aliases:
            return std
    return None


def parse_finstate_dataframe(df: pd.DataFrame) -> dict[str, float]:
    """
    `OpenDartReader.finstate()` 가 반환하는 DataFrame 을
    {계정표준명: 금액} dict 로 정리.

    - 연결재무제표(CFS) 우선, 없으면 별도(OFS).
    - 단위: 원.
    """
    if df is None or df.empty:
        return {}

    # 연결재무제표(fs_div == 'CFS') 우선
    if "fs_div" in df.columns:
        cfs = df[df["fs_div"] == "CFS"]
        if not cfs.empty:
            df = cfs

    out: dict[str, float] = {}
    for _, row in df.iterrows():
        std = _normalize_account_name(str(row.get("account_nm", "")))
        if std is None:
            continue
        # thstrm_amount: 당기금액 (문자열, 쉼표 포함 가능)
        raw = row.get("thstrm_amount", "")
        try:
            value = float(str(raw).replace(",", ""))
        except (ValueError, TypeError):
            continue
        out[std] = value
    return out


def load_financials(
    stock_code: str,
    year: int,
    report_type: str = "annual",
    use_cache: bool = True,
) -> dict[str, float]:
    """
    한 종목의 한 해 재무제표 핵심 항목을 dict 로 반환.

    Parameters
    ----------
    stock_code : str
        6자리 종목코드 (예: "005930" 삼성전자).
    year : int
        조회 연도 (예: 2024).
    report_type : str
        "annual" = 사업보고서(11011) — Phase 2 기본값.
        "q3"     = 3분기보고서(11014)
        "h1"     = 반기보고서(11012)
        "q1"     = 1분기보고서(11013)

    Returns
    -------
    dict
        {"revenue": ..., "op_profit": ..., "net_income": ...,
         "total_assets": ..., "total_debt": ..., "total_equity": ...}
        없으면 빈 dict.
    """
    cache_key = f"dart_{stock_code}_{year}_{report_type}"
    ttl = load_thresholds()["cache"]["fundamental_ttl_days"]

    if use_cache and cache.is_fresh(cache_key, ttl):
        df = cache.load(cache_key)
        if df is not None and not df.empty:
            logger.info(f"[캐시] DART {stock_code} {year}")
            return parse_finstate_dataframe(df)

    client = _client()
    logger.info(f"[API] DART {stock_code} {year} {report_type}")

    # OpenDartReader.finstate(종목코드_or_회사명, 사업연도, [분기/반기])
    # report_type 이 "annual" 이면 두 번째 인자만으로 사업보고서
    if report_type == "annual":
        df = client.finstate(stock_code, year)
    else:
        # OpenDartReader의 reprt_code 매핑: 11013(1Q), 11012(H1), 11014(3Q)
        kind_map = {"q1": "1Q", "h1": "H1", "q3": "3Q"}
        df = client.finstate(stock_code, year, reprt_code=kind_map.get(report_type, "annual"))

    if df is None or df.empty:
        logger.warning(f"DART 응답 비어있음: {stock_code} {year} {report_type}")
        return {}

    cache.save(cache_key, df)
    return parse_finstate_dataframe(df)


def load_universe_financials(
    codes: Iterable[str],
    year: int,
    report_type: str = "annual",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    여러 종목의 재무제표를 한 번에 받아 DataFrame 으로 반환.

    columns = [code, revenue, op_profit, net_income, total_assets, total_debt, total_equity]
    실패한 종목은 NaN 으로 채움 (PROJECT_SPEC.md §7.3).
    """
    rows: list[dict[str, Any]] = []
    for code in codes:
        try:
            fin = load_financials(code, year, report_type, use_cache)
            fin["code"] = code
            rows.append(fin)
        except DartKeyMissingError:
            raise  # 키 없음은 치명적, 바로 던짐
        except Exception as e:  # noqa: BLE001
            logger.error(f"DART 로드 실패 {code}: {e}")
            rows.append({"code": code})

    df = pd.DataFrame(rows)
    # code 컬럼을 인덱스로
    if "code" in df.columns:
        df = df.set_index("code")
    return df
