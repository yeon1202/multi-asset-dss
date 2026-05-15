"""
캐시 모듈 단위 테스트.

tmp_path: pytest 내장 fixture — 테스트마다 격리된 임시 디렉토리를 줍니다.
monkeypatch: 런타임에 변수·속성을 임시로 바꿔주는 fixture.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.data import cache


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path: Path):
    """CACHE_DIR 를 임시 폴더로 갈아끼움 → 실제 data/ 폴더 오염 방지."""
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    return tmp_path


def _sample_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame({"Close": [100, 101, 102, 103, 104]}, index=idx)


def test_save_and_load_roundtrip(isolated_cache):
    df = _sample_df()
    cache.save("test_key", df)
    loaded = cache.load("test_key")
    assert loaded is not None
    assert list(loaded["Close"]) == [100, 101, 102, 103, 104]


def test_load_missing_returns_none(isolated_cache):
    assert cache.load("nonexistent") is None


def test_is_fresh_false_when_missing(isolated_cache):
    assert cache.is_fresh("nope", ttl_days=1) is False


def test_is_fresh_true_after_save(isolated_cache):
    cache.save("k", _sample_df())
    assert cache.is_fresh("k", ttl_days=1) is True


def test_is_fresh_false_when_stale(isolated_cache):
    path = cache.save("old", _sample_df())
    # 파일 mtime 을 8일 전으로 강제 변경
    old_time = (datetime.now() - timedelta(days=8)).timestamp()
    import os
    os.utime(path, (old_time, old_time))
    assert cache.is_fresh("old", ttl_days=7) is False


def test_clear_specific_key(isolated_cache):
    cache.save("a", _sample_df())
    cache.save("b", _sample_df())
    removed = cache.clear("a")
    assert removed == 1
    assert cache.load("a") is None
    assert cache.load("b") is not None


def test_clear_all(isolated_cache):
    cache.save("a", _sample_df())
    cache.save("b", _sample_df())
    removed = cache.clear()
    assert removed == 2


def test_unsafe_key_is_sanitized(isolated_cache):
    """슬래시 들어간 키도 파일로 저장 가능해야 함."""
    cache.save("a/b", _sample_df())
    assert cache.load("a/b") is not None
    assert (isolated_cache / "a_b.csv").exists()
