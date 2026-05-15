"""
거시 데이터 로더 — Phase 3.

소스:
  - ECOS (한국은행 경제통계시스템) → 기준금리, 환율, CPI
    https://ecos.bok.or.kr/api
  - FRED (St. Louis Fed)            → VIX, 미국 국채금리, 신용 스프레드
    https://fred.stlouisfed.org/docs/api/

키는 .env 의 ECOS_API_KEY / FRED_API_KEY 에서 로드 (PROJECT_SPEC.md §8).
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger

from src.data import cache
from src.utils.config_loader import PROJECT_ROOT, load_macro_config, load_thresholds

# .env 로드
load_dotenv(PROJECT_ROOT / ".env")


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------
class EcosKeyMissingError(RuntimeError):
    """ECOS_API_KEY 가 환경변수에 없을 때."""


class FredKeyMissingError(RuntimeError):
    """FRED_API_KEY 가 환경변수에 없을 때."""


def _get_key(env_var: str, error_cls: type[RuntimeError]) -> str:
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise error_cls(
            f"{env_var} 가 설정되지 않았습니다.\n"
            f"1) 무료 발급 — README 의 'API 키 설정' 참조\n"
            f"2) {PROJECT_ROOT}/.env 파일에 '{env_var}=발급받은키' 추가"
        )
    return key


# ----------------------------------------------------------------------
# ECOS
# ----------------------------------------------------------------------
def _parse_ecos_time(time_str: str, cycle: str) -> pd.Timestamp:
    """ECOS TIME 문자열을 cycle 에 맞춰 Timestamp 로 변환."""
    try:
        if cycle == "D":
            return pd.to_datetime(time_str, format="%Y%m%d")
        if cycle == "M":
            return pd.to_datetime(time_str, format="%Y%m")
        if cycle == "Y":
            return pd.to_datetime(time_str, format="%Y")
        if cycle == "Q":
            # YYYYQn 형식 — 분기 첫달 1일로
            year = int(time_str[:4])
            q = int(time_str[5])
            return pd.Timestamp(year=year, month=(q - 1) * 3 + 1, day=1)
    except (ValueError, TypeError, IndexError):
        return pd.NaT
    return pd.NaT


def _ecos_date_range(cycle: str, lookback_months: int) -> tuple[str, str]:
    """ECOS 가 받는 YYYYMM(M주기) 또는 YYYYMMDD(D주기) 시작·끝 문자열."""
    today = date.today()
    # 시작일을 lookback_months 만큼 과거로
    months_back = lookback_months
    start = today.replace(day=1) - timedelta(days=months_back * 31)
    if cycle == "M":
        return start.strftime("%Y%m"), today.strftime("%Y%m")
    if cycle == "D":
        return start.strftime("%Y%m%d"), today.strftime("%Y%m%d")
    if cycle == "Y":
        return start.strftime("%Y"), today.strftime("%Y")
    if cycle == "Q":
        # ECOS 분기 표기: YYYYQn (예: 2024Q1)
        q_start = (start.month - 1) // 3 + 1
        q_end = (today.month - 1) // 3 + 1
        return f"{start.year}Q{q_start}", f"{today.year}Q{q_end}"
    raise ValueError(f"지원하지 않는 cycle: {cycle}")


def load_ecos_series(
    name: str,
    use_cache: bool = True,
) -> pd.Series:
    """
    ECOS 시리즈 한 개 로드 (config/macro.yaml 의 ecos.series.<name>).

    Returns
    -------
    pd.Series
        index = pd.DatetimeIndex (cycle 에 따라 일/월/분기),
        name = <name>, dtype = float.
    """
    cfg = load_macro_config()["ecos"]
    if name not in cfg["series"]:
        raise KeyError(f"ECOS 시리즈 '{name}' 가 config/macro.yaml 에 없음")
    spec = cfg["series"][name]
    code = spec["code"]
    cycle = spec["cycle"]
    item = spec.get("item", "0")
    start, end = _ecos_date_range(cycle, cfg["lookback_months"])

    cache_key = f"ecos_{name}_{cycle}_{start}_{end}"
    ttl = load_thresholds()["cache"]["price_ttl_days"]

    if use_cache and cache.is_fresh(cache_key, ttl):
        df = cache.load(cache_key)
        if df is not None and not df.empty:
            logger.info(f"[캐시] ECOS {name}")
            return df.iloc[:, 0].rename(name)

    api_key = _get_key("ECOS_API_KEY", EcosKeyMissingError)
    url = (
        f"{cfg['base_url']}/{api_key}/json/kr/1/100000/"
        f"{code}/{cycle}/{start}/{end}/{item}"
    )
    logger.info(f"[API] ECOS {name} ({cycle}, {start}~{end})")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    # ECOS 응답 구조: {"StatisticSearch": {"row": [{...}, ...], "list_total_count": N}}
    # 또는 에러: {"RESULT": {"CODE": "...", "MESSAGE": "..."}}
    if "RESULT" in payload:
        msg = payload["RESULT"].get("MESSAGE", "")
        raise RuntimeError(f"ECOS API 오류 ({name}): {msg}")
    rows = payload.get("StatisticSearch", {}).get("row", [])
    if not rows:
        raise ValueError(f"ECOS {name} 응답에 데이터가 없음")

    df = pd.DataFrame(rows)
    df["TIME"] = df["TIME"].apply(lambda t: _parse_ecos_time(t, cycle))
    df["DATA_VALUE"] = pd.to_numeric(df["DATA_VALUE"], errors="coerce")
    df = df.dropna(subset=["TIME", "DATA_VALUE"]).set_index("TIME").sort_index()
    series = df["DATA_VALUE"].rename(name)

    cache.save(cache_key, series.to_frame())
    return series


# ----------------------------------------------------------------------
# FRED
# ----------------------------------------------------------------------
def load_fred_series(
    name: str,
    use_cache: bool = True,
) -> pd.Series:
    """
    FRED 시리즈 한 개 로드 (config/macro.yaml 의 fred.series.<name>).

    Returns
    -------
    pd.Series
        index = pd.DatetimeIndex (daily), name = <name>, dtype = float.
    """
    cfg = load_macro_config()["fred"]
    if name not in cfg["series"]:
        raise KeyError(f"FRED 시리즈 '{name}' 가 config/macro.yaml 에 없음")
    series_id = cfg["series"][name]["id"]
    lookback = cfg["lookback_days"]
    end = date.today()
    start = end - timedelta(days=lookback)

    cache_key = f"fred_{name}_{start}_{end}"
    ttl = load_thresholds()["cache"]["price_ttl_days"]

    if use_cache and cache.is_fresh(cache_key, ttl):
        df = cache.load(cache_key)
        if df is not None and not df.empty:
            logger.info(f"[캐시] FRED {name}")
            return df.iloc[:, 0].rename(name)

    api_key = _get_key("FRED_API_KEY", FredKeyMissingError)
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start.isoformat(),
        "observation_end": end.isoformat(),
    }
    logger.info(f"[API] FRED {name} ({series_id})")
    resp = requests.get(cfg["base_url"], params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    obs = payload.get("observations", [])
    if not obs:
        raise ValueError(f"FRED {name} 응답에 데이터가 없음")

    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"])
    # FRED 결측은 "." 으로 옴
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).set_index("date").sort_index()
    series = df["value"].rename(name)

    cache.save(cache_key, series.to_frame())
    return series


# ----------------------------------------------------------------------
# 통합 로드
# ----------------------------------------------------------------------
def load_all_macro(use_cache: bool = True) -> pd.DataFrame:
    """
    config/macro.yaml 에 정의된 ECOS + FRED 시리즈 전체를 daily-aligned DataFrame 으로 반환.

    각 시리즈는 ffill 로 일별 정렬 (월별 데이터는 다음 갱신 전까지 같은 값 유지).

    Returns
    -------
    pd.DataFrame
        index = DatetimeIndex (영업일 기준일 정렬, daily),
        columns = ECOS·FRED feature 이름들.
    """
    cfg = load_macro_config()
    series_list: list[pd.Series] = []

    for name in cfg["ecos"]["series"]:
        try:
            s = load_ecos_series(name, use_cache=use_cache)
            series_list.append(s)
        except (EcosKeyMissingError, FredKeyMissingError):
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning(f"ECOS {name} 로드 실패 — 스킵: {e}")

    for name in cfg["fred"]["series"]:
        try:
            s = load_fred_series(name, use_cache=use_cache)
            series_list.append(s)
        except (EcosKeyMissingError, FredKeyMissingError):
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning(f"FRED {name} 로드 실패 — 스킵: {e}")

    if not series_list:
        raise RuntimeError("로드된 거시 시리즈가 하나도 없습니다.")

    # 일별 인덱스로 정렬 후 ffill
    merged = pd.concat(series_list, axis=1).sort_index()
    # 영업일 기준 (B): 주말 제거 후 결측은 ffill
    daily = merged.asfreq("D").ffill()
    return daily
