"""
설정 로더가 universe.yaml / thresholds.yaml 을 잘 읽는지 확인.
"""
from __future__ import annotations

from src.utils.config_loader import load_thresholds, load_universe


def test_universe_has_phase1_assets():
    u = load_universe()
    codes = {a["code"] for a in u["assets"]}
    # PROJECT_SPEC.md §6 Phase 1 6개 자산 모두 포함
    expected = {"069500", "360750", "133690", "148070", "132030", "261240"}
    assert expected.issubset(codes)


def test_thresholds_have_required_keys():
    t = load_thresholds()
    assert t["rsi"]["period"] >= 1
    assert 0 < t["rsi"]["oversold"] < t["rsi"]["overbought"] < 100
    assert t["moving_average"]["short"] < t["moving_average"]["long"]
    assert t["volatility"]["annualize_factor"] > 0
    assert t["cache"]["price_ttl_days"] >= 1


def test_load_universe_is_cached():
    """lru_cache 가 동작하면 두 번 호출해도 같은 객체."""
    a = load_universe()
    b = load_universe()
    assert a is b
