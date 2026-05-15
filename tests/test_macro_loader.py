"""
거시 데이터 로더 테스트 — 실제 HTTP 호출은 monkeypatch 로 차단.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import cache, macro_loader


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def clear_yaml_cache():
    """yaml lru_cache 가 테스트 간 상태를 공유하지 않게."""
    from src.utils.config_loader import load_yaml
    load_yaml.cache_clear()


# ---------- API key handling ----------

def test_ecos_key_missing_raises(monkeypatch):
    monkeypatch.delenv("ECOS_API_KEY", raising=False)
    with pytest.raises(macro_loader.EcosKeyMissingError):
        macro_loader._get_key("ECOS_API_KEY", macro_loader.EcosKeyMissingError)


def test_fred_key_missing_raises(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(macro_loader.FredKeyMissingError):
        macro_loader._get_key("FRED_API_KEY", macro_loader.FredKeyMissingError)


def test_get_key_present(monkeypatch):
    monkeypatch.setenv("ECOS_API_KEY", "test-key-1234")
    assert macro_loader._get_key("ECOS_API_KEY", macro_loader.EcosKeyMissingError) == "test-key-1234"


# ---------- ECOS ----------

class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_load_ecos_series_parses_response(isolated_cache, monkeypatch):
    monkeypatch.setenv("ECOS_API_KEY", "test")
    payload = {
        "StatisticSearch": {
            "list_total_count": 3,
            "row": [
                {"TIME": "202401", "DATA_VALUE": "3.50"},
                {"TIME": "202402", "DATA_VALUE": "3.50"},
                {"TIME": "202403", "DATA_VALUE": "3.25"},
            ],
        }
    }
    monkeypatch.setattr(macro_loader.requests, "get",
                         lambda *a, **kw: FakeResponse(payload))

    s = macro_loader.load_ecos_series("base_rate", use_cache=False)
    assert len(s) == 3
    assert s.iloc[-1] == pytest.approx(3.25)
    assert s.name == "base_rate"


def test_load_ecos_series_error_payload(isolated_cache, monkeypatch):
    """ECOS 가 RESULT 키로 에러 반환 시 RuntimeError."""
    monkeypatch.setenv("ECOS_API_KEY", "test")
    payload = {"RESULT": {"CODE": "INFO-200", "MESSAGE": "해당하는 데이터가 없습니다."}}
    monkeypatch.setattr(macro_loader.requests, "get",
                         lambda *a, **kw: FakeResponse(payload))
    with pytest.raises(RuntimeError, match="해당하는 데이터가 없습니다"):
        macro_loader.load_ecos_series("base_rate", use_cache=False)


def test_load_ecos_unknown_series_raises():
    with pytest.raises(KeyError, match="없음"):
        macro_loader.load_ecos_series("nonexistent_series_xyz", use_cache=False)


# ---------- FRED ----------

def test_load_fred_series_parses_response(isolated_cache, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test")
    payload = {
        "observations": [
            {"date": "2024-01-02", "value": "12.34"},
            {"date": "2024-01-03", "value": "13.10"},
            {"date": "2024-01-04", "value": "."},  # FRED 결측치
            {"date": "2024-01-05", "value": "12.80"},
        ]
    }
    monkeypatch.setattr(macro_loader.requests, "get",
                         lambda *a, **kw: FakeResponse(payload))
    s = macro_loader.load_fred_series("vix", use_cache=False)
    assert len(s) == 3  # "." 은 제거
    assert s.iloc[0] == pytest.approx(12.34)
    assert s.iloc[-1] == pytest.approx(12.80)


def test_load_fred_unknown_series_raises():
    with pytest.raises(KeyError, match="없음"):
        macro_loader.load_fred_series("not_a_series", use_cache=False)


# ---------- date range ----------

def test_ecos_date_range_monthly():
    start, end = macro_loader._ecos_date_range("M", 12)
    # YYYYMM 형태
    assert len(start) == 6 and start.isdigit()
    assert len(end) == 6 and end.isdigit()
    assert start < end


def test_ecos_date_range_daily():
    start, end = macro_loader._ecos_date_range("D", 12)
    assert len(start) == 8 and len(end) == 8


def test_ecos_date_range_invalid_cycle():
    with pytest.raises(ValueError):
        macro_loader._ecos_date_range("X", 12)
