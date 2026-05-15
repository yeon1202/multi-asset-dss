"""
기술적 지표 (Technical Indicators) — Phase 1.

- RSI (Relative Strength Index): 상대강도지수, 0~100. 과매수/과매도 판단.
- 단순이동평균 (SMA): 단기·장기 추세
- 변동성 (Volatility): 일일 수익률의 표준편차 (연환산)

모든 입력은 종가(Series) 1차원. NaN 입력에 대해서는 NaN을 그대로 반환합니다.
PROJECT_SPEC.md §7.3 결측치를 0으로 채우지 않음.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder의 RSI — `pandas-ta` 등과 동일한 표준 정의.

    RSI = 100 - 100 / (1 + RS),  RS = 평균상승폭 / 평균하락폭

    Parameters
    ----------
    close : pd.Series
        종가 시계열.
    period : int
        평균을 낼 기간 (보통 14일).

    Returns
    -------
    pd.Series
        같은 index, 0~100 범위의 RSI 값.
    """
    if period <= 0:
        raise ValueError("period 는 1 이상이어야 합니다.")
    if len(close) < period + 1:
        # 데이터가 부족하면 전부 NaN 반환
        return pd.Series([np.nan] * len(close), index=close.index, name="RSI")

    # diff(): 이전 행과의 차이
    delta = close.diff()
    gain = delta.clip(lower=0)   # 음수는 0으로 (상승분만)
    loss = -delta.clip(upper=0)  # 양수는 0으로 후 부호 반전 (하락분만, 양수)

    # Wilder smoothing = ewm with alpha=1/period, adjust=False
    # ewm: exponentially weighted moving average (지수가중 이동평균)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    # 분모 0 처리: 손실이 전혀 없으면 RSI=100
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    rsi_series = rsi_series.where(avg_loss != 0, 100)
    rsi_series.name = "RSI"
    return rsi_series


def sma(close: pd.Series, window: int) -> pd.Series:
    """단순이동평균 (Simple Moving Average)."""
    if window <= 0:
        raise ValueError("window 는 1 이상이어야 합니다.")
    # rolling(window).mean() → 슬라이딩 윈도우 평균
    return close.rolling(window=window, min_periods=window).mean().rename(f"SMA_{window}")


def daily_returns(close: pd.Series) -> pd.Series:
    """일간 수익률(단순). 첫 행은 NaN."""
    # pct_change(): (오늘/어제) - 1
    return close.pct_change().rename("ret")


def volatility(
    close: pd.Series,
    window: int = 20,
    annualize_factor: int = 252,
) -> pd.Series:
    """
    연환산 변동성 = 일간 수익률의 표준편차 × √(영업일/년).

    한국·미국 모두 연 영업일 ≈ 252. annualize_factor 로 변경 가능.
    """
    if window <= 1:
        raise ValueError("window 는 2 이상이어야 합니다.")
    ret = daily_returns(close)
    # std(ddof=1): 표본 표준편차 (n-1 분모) — 통계학 표준
    vol = ret.rolling(window=window, min_periods=window).std(ddof=1)
    return (vol * np.sqrt(annualize_factor)).rename(f"VOL_{window}")


def summarize(
    close: pd.Series,
    rsi_period: int = 14,
    ma_short: int = 20,
    ma_long: int = 60,
    vol_window: int = 20,
    annualize_factor: int = 252,
) -> dict[str, float]:
    """
    가장 최근 시점의 지표 값 한 묶음.

    대시보드 표에 띄울 용도. NaN 이면 그대로 NaN 으로 둡니다.
    """
    r = rsi(close, rsi_period).iloc[-1]
    s = sma(close, ma_short).iloc[-1]
    l = sma(close, ma_long).iloc[-1]
    v = volatility(close, vol_window, annualize_factor).iloc[-1]
    last = close.iloc[-1]
    return {
        "last_close": float(last) if pd.notna(last) else float("nan"),
        "rsi": float(r) if pd.notna(r) else float("nan"),
        f"sma_{ma_short}": float(s) if pd.notna(s) else float("nan"),
        f"sma_{ma_long}": float(l) if pd.notna(l) else float("nan"),
        "vol_annualized": float(v) if pd.notna(v) else float("nan"),
    }
