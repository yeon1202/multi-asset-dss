"""
데이터 캐싱 모듈.

같은 API를 하루에 두 번 호출하지 않도록 디스크에 CSV로 저장하고,
다음 호출 때 TTL(유효기간) 내라면 디스크에서 읽습니다.

PROJECT_SPEC.md §7.1 데이터 정책의 '캐싱' 규칙을 구현.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.utils.config_loader import PROJECT_ROOT

# 캐시 저장소: data/raw/
CACHE_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(key: str) -> Path:
    """캐시 키를 안전한 파일 경로로 변환."""
    # 종목코드/티커에 / 가 있으면 _ 로 치환 (파일명 안전)
    safe = key.replace("/", "_").replace("\\", "_")
    return CACHE_DIR / f"{safe}.csv"


def is_fresh(key: str, ttl_days: int) -> bool:
    """캐시가 ttl_days 이내에 갱신되었는지."""
    path = _cache_path(key)
    if not path.exists():
        return False
    # 파일 수정 시각을 datetime으로
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return datetime.now() - mtime < timedelta(days=ttl_days)


def load(key: str) -> pd.DataFrame | None:
    """캐시에서 DataFrame 로드. 없으면 None.

    인덱스가 ISO 날짜 형태이면 DatetimeIndex 로 변환,
    아니면(종목코드 등) 그대로 유지. parse_dates=True 의
    포맷 추측 워닝을 피하기 위해 명시적으로 처리.
    """
    path = _cache_path(key)
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    try:
        # format="ISO8601": YYYY-MM-DD 같은 표준 날짜만 허용
        df.index = pd.to_datetime(df.index, format="ISO8601")
    except (ValueError, TypeError):
        pass  # 날짜 인덱스가 아니면 그대로
    return df


def save(key: str, df: pd.DataFrame) -> Path:
    """DataFrame을 캐시로 저장."""
    path = _cache_path(key)
    df.to_csv(path, encoding="utf-8")
    return path


def clear(key: str | None = None) -> int:
    """캐시 삭제. key=None 이면 전체. 삭제 개수 반환."""
    if key is None:
        files = list(CACHE_DIR.glob("*.csv"))
    else:
        f = _cache_path(key)
        files = [f] if f.exists() else []
    for f in files:
        f.unlink()  # 파일 삭제
    return len(files)
