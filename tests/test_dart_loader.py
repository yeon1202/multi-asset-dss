"""
DART 로더 단위 테스트.

실제 API 호출 없이 동작 검증을 위해 OpenDartReader 클라이언트를 mock 으로 치환.
파싱·캐싱·키 처리 로직만 검증.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import cache, dart_loader


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def sample_finstate_df() -> pd.DataFrame:
    """OpenDartReader.finstate() 가 돌려주는 형태의 미니 DataFrame."""
    return pd.DataFrame(
        {
            "fs_div": ["CFS"] * 6,
            "account_nm": [
                "매출액", "영업이익", "당기순이익",
                "자산총계", "부채총계", "자본총계",
            ],
            "thstrm_amount": [
                "300,870,903", "26,969,099", "26,433,925",
                "455,906,490", "115,950,373", "339,956,117",
            ],
        }
    )


def test_parse_finstate_dataframe_basic(sample_finstate_df):
    out = dart_loader.parse_finstate_dataframe(sample_finstate_df)
    assert out["revenue"] == 300_870_903
    assert out["op_profit"] == 26_969_099
    assert out["net_income"] == 26_433_925
    assert out["total_assets"] == 455_906_490
    assert out["total_debt"] == 115_950_373
    assert out["total_equity"] == 339_956_117


def test_parse_finstate_dataframe_empty():
    assert dart_loader.parse_finstate_dataframe(pd.DataFrame()) == {}
    assert dart_loader.parse_finstate_dataframe(None) == {}  # type: ignore[arg-type]


def test_parse_finstate_dataframe_prefers_cfs():
    """별도(OFS)와 연결(CFS)이 같이 오면 CFS 우선."""
    df = pd.DataFrame(
        {
            "fs_div": ["OFS", "CFS"],
            "account_nm": ["매출액", "매출액"],
            "thstrm_amount": ["100", "999"],
        }
    )
    out = dart_loader.parse_finstate_dataframe(df)
    assert out["revenue"] == 999


def test_parse_finstate_ignores_unknown_account():
    df = pd.DataFrame(
        {
            "fs_div": ["CFS", "CFS"],
            "account_nm": ["매출액", "기타포괄손익"],  # 두 번째는 미사용
            "thstrm_amount": ["100", "5"],
        }
    )
    out = dart_loader.parse_finstate_dataframe(df)
    assert out == {"revenue": 100}


def test_parse_finstate_skips_unparsable_amount():
    df = pd.DataFrame(
        {
            "fs_div": ["CFS", "CFS"],
            "account_nm": ["매출액", "영업이익"],
            "thstrm_amount": ["abc", "200"],
        }
    )
    out = dart_loader.parse_finstate_dataframe(df)
    assert out == {"op_profit": 200}


def test_get_api_key_missing_raises(monkeypatch):
    monkeypatch.delenv("DART_API_KEY", raising=False)
    with pytest.raises(dart_loader.DartKeyMissingError):
        dart_loader.get_api_key()


def test_get_api_key_present(monkeypatch):
    monkeypatch.setenv("DART_API_KEY", "fake-key-1234")
    assert dart_loader.get_api_key() == "fake-key-1234"


def test_load_financials_uses_cache_when_fresh(
    isolated_cache, sample_finstate_df, monkeypatch
):
    """캐시가 신선하면 클라이언트 호출 없이 디스크에서 반환."""
    cache_key = "dart_005930_2024_annual"
    cache.save(cache_key, sample_finstate_df)

    # 클라이언트가 호출되면 에러 — 호출 자체가 실패여야 함
    def boom(*a, **kw):
        raise AssertionError("API 호출이 발생하면 안 됨")

    monkeypatch.setattr(dart_loader, "_client", boom)
    out = dart_loader.load_financials("005930", 2024)
    assert out["revenue"] == 300_870_903


def test_load_financials_calls_client_when_no_cache(
    isolated_cache, sample_finstate_df, monkeypatch
):
    """캐시가 없으면 client.finstate() 호출 → 결과를 캐시에 저장."""

    class FakeClient:
        def finstate(self, code, year, reprt_code=None):
            return sample_finstate_df

    monkeypatch.setattr(dart_loader, "_client", lambda: FakeClient())
    out = dart_loader.load_financials("005930", 2024, use_cache=False)
    assert out["revenue"] == 300_870_903
    # 캐시 파일이 새로 생성되었는지
    assert (isolated_cache / "dart_005930_2024_annual.csv").exists()


def test_load_universe_financials_aggregates(
    isolated_cache, sample_finstate_df, monkeypatch
):
    """3종목 — 그 중 하나는 빈 응답 → DataFrame 인덱스에 모두 포함, NaN 처리."""

    call_count = {"n": 0}

    class FakeClient:
        def finstate(self, code, year, reprt_code=None):
            call_count["n"] += 1
            if code == "BAD":
                return pd.DataFrame()  # 빈 응답
            return sample_finstate_df

    monkeypatch.setattr(dart_loader, "_client", lambda: FakeClient())
    df = dart_loader.load_universe_financials(
        ["005930", "BAD", "000660"], year=2024, use_cache=False
    )
    assert set(df.index) == {"005930", "BAD", "000660"}
    # 정상 종목은 값 있음
    assert df.loc["005930", "revenue"] == 300_870_903
    # 빈 응답 종목은 NaN
    assert pd.isna(df.loc["BAD", "revenue"])
