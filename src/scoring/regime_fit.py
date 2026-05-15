"""
자산별 레짐 적합도 점수 — Phase 3.

각 자산은 macro.yaml 의 `asset_preference` 에서 [-1, +1] 선호도를 가짐:
  +1 → 강한 risk-on 자산 (성장주, 코스닥, 신흥국 등)
  -1 → 강한 risk-off 자산 (국채, 금, 현금성)
   0 → 중립

레짐 점수 r ∈ [-1, +1] 일 때 자산 적합도:

       fit = 50 + 50 · pref · r

  - pref=+1, r=+1  → 100 (perfect match — risk-on 자산 × risk-on 레짐)
  - pref=+1, r=-1  →   0 (worst — risk-on 자산 × risk-off 레짐)
  - pref=-1, r=-1  → 100 (risk-off 자산 × risk-off 레짐)
  - pref=0          →  50 (중립 자산은 항상 50)
"""
from __future__ import annotations

import math
from typing import Mapping

import pandas as pd


def asset_fit_score(preference: float, regime_score: float) -> float:
    """단일 자산의 레짐 적합도 [0, 100]."""
    if preference is None or regime_score is None:
        return float("nan")
    if isinstance(preference, float) and math.isnan(preference):
        return float("nan")
    if isinstance(regime_score, float) and math.isnan(regime_score):
        return float("nan")
    # pref, regime 모두 [-1, 1] 가정 → 곱 ∈ [-1, 1] → fit ∈ [0, 100]
    return 50.0 + 50.0 * float(preference) * float(regime_score)


def asset_fit_table(
    preferences: Mapping[str, float],
    regime_score: float,
) -> pd.DataFrame:
    """
    전체 자산의 적합도 표.

    Parameters
    ----------
    preferences : dict
        {"069500": 1.0, "148070": -0.7, ...}
    regime_score : float
        현재 레짐 종합 점수.

    Returns
    -------
    pd.DataFrame
        index = code, columns = [preference, regime_fit].
    """
    rows = []
    for code, pref in preferences.items():
        rows.append({
            "code": code,
            "preference": pref,
            "regime_fit": asset_fit_score(pref, regime_score),
        })
    return pd.DataFrame(rows).set_index("code")
