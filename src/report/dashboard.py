"""
Phase 1 대시보드 — Streamlit.

실행:  `streamlit run src/report/dashboard.py`

표시 내용:
  1. 자산별 정규화 가격 차트 (시작=100 으로 맞춰서 상대 성과 비교)
  2. 자산별 현재 RSI / SMA20 / SMA60 / 연환산 변동성 표
  3. 가격 vs 이동평균 비교 차트 (자산별)

PROJECT_SPEC.md §2.2: 분석·추천만, 자동 주문은 안 합니다.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Streamlit이 이 파일을 직접 실행할 때 프로젝트 루트를 import path에 추가
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.price_loader import (
    get_close_matrix,
    load_universe_prices,
)
from src.indicators.technical import rsi, sma, summarize, volatility
from src.utils.config_loader import load_thresholds, load_universe

# ----------------------------------------------------------------------
# 페이지 설정
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="다자산 모니터 — Phase 1",
    page_icon="📈",
    layout="wide",
)

# 면책 문구 (PROJECT_SPEC.md §9.3)
st.info(
    "ℹ️ 이 리포트는 **정보 제공 목적**이며, 투자 결정과 그 결과는 본인 책임입니다. "
    "자동 주문은 수행하지 않습니다."
)

st.title("📈 다자산 모니터")
st.caption("Phase 1 · 한국·미국 ETF + 금 + 채권 + 달러 6개 자산")


# ----------------------------------------------------------------------
# 데이터 로드 — Streamlit 캐시로 화면 새로고침 시 중복 호출 방지
# ----------------------------------------------------------------------
@st.cache_data(ttl=60 * 60 * 6)  # 6시간
def _load_all(lookback_days: int) -> dict[str, pd.DataFrame]:
    return load_universe_prices(lookback_days=lookback_days)


with st.sidebar:
    st.header("⚙️ 설정")
    universe = load_universe()
    default_days = universe.get("default_lookback_days", 365)
    lookback = st.slider(
        "조회 기간 (일)",
        min_value=90,
        max_value=730,
        value=default_days,
        step=30,
    )
    thresholds = load_thresholds()
    st.caption(
        f"RSI: {thresholds['rsi']['period']}일 / "
        f"이평: {thresholds['moving_average']['short']}·"
        f"{thresholds['moving_average']['long']}일 / "
        f"변동성 윈도우: {thresholds['volatility']['window']}일"
    )

with st.spinner("📡 데이터 다운로드 중... (캐시되면 다음부터 즉시 로드)"):
    prices = _load_all(lookback)

if not prices:
    st.error("데이터를 하나도 로드하지 못했습니다. 인터넷 연결을 확인해주세요.")
    st.stop()

# 이름 매핑
name_by_code: dict[str, str] = {a["code"]: a["name"] for a in universe["assets"]}
category_by_code: dict[str, str] = {a["code"]: a["category"] for a in universe["assets"]}

# ----------------------------------------------------------------------
# 1) 정규화 가격 차트 (시작점 = 100)
# ----------------------------------------------------------------------
st.subheader("1️⃣ 정규화 가격 (시작=100)")
st.caption("같은 출발선에서 누가 더 잘 갔는지 한눈에 비교")

closes = get_close_matrix(prices)
# 첫 유효값으로 나눠 100을 곱함 → 상대 성과
normalized = closes.apply(lambda s: s / s.dropna().iloc[0] * 100)
normalized = normalized.rename(columns=name_by_code)

fig_norm = px.line(
    normalized,
    labels={"value": "정규화 가격 (시작=100)", "Date": "날짜", "variable": "자산"},
)
fig_norm.update_layout(legend_title="자산", height=450)
st.plotly_chart(fig_norm, use_container_width=True)

# ----------------------------------------------------------------------
# 2) 최신 지표 요약 표
# ----------------------------------------------------------------------
st.subheader("2️⃣ 최신 지표 요약")

rsi_period = thresholds["rsi"]["period"]
ma_short = thresholds["moving_average"]["short"]
ma_long = thresholds["moving_average"]["long"]
vol_window = thresholds["volatility"]["window"]
ann_factor = thresholds["volatility"]["annualize_factor"]
ob = thresholds["rsi"]["overbought"]
os_ = thresholds["rsi"]["oversold"]

rows = []
for code, df in prices.items():
    s = summarize(
        df["Close"],
        rsi_period=rsi_period,
        ma_short=ma_short,
        ma_long=ma_long,
        vol_window=vol_window,
        annualize_factor=ann_factor,
    )
    last_close = s["last_close"]
    sma_s = s[f"sma_{ma_short}"]
    sma_l = s[f"sma_{ma_long}"]
    # 추세 신호: 종가 vs 이평선
    trend = "🟢 강세" if last_close > sma_l else "🔴 약세"
    # RSI 신호
    r = s["rsi"]
    if pd.isna(r):
        rsi_sig = "—"
    elif r >= ob:
        rsi_sig = f"🔴 과매수 ({r:.1f})"
    elif r <= os_:
        rsi_sig = f"🟢 과매도 ({r:.1f})"
    else:
        rsi_sig = f"⚪ 중립 ({r:.1f})"

    rows.append(
        {
            "종목": name_by_code.get(code, code),
            "카테고리": category_by_code.get(code, "-"),
            "종가": f"{last_close:,.0f}" if pd.notna(last_close) else "—",
            "RSI 신호": rsi_sig,
            f"SMA{ma_short}": f"{sma_s:,.0f}" if pd.notna(sma_s) else "—",
            f"SMA{ma_long}": f"{sma_l:,.0f}" if pd.notna(sma_l) else "—",
            "추세": trend,
            "연환산 변동성": f"{s['vol_annualized']*100:.1f}%" if pd.notna(s["vol_annualized"]) else "—",
        }
    )

summary_df = pd.DataFrame(rows)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.caption(
    f"RSI ≥ {ob}: 과매수 · RSI ≤ {os_}: 과매도 · "
    f"추세 = 종가 > SMA{ma_long} 이면 강세"
)

# ----------------------------------------------------------------------
# 3) 자산별 상세 차트 (가격 + 이평선 + RSI)
# ----------------------------------------------------------------------
st.subheader("3️⃣ 자산별 상세")

selected_code = st.selectbox(
    "자산 선택",
    options=list(prices.keys()),
    format_func=lambda c: f"{name_by_code.get(c, c)} ({c})",
)
df = prices[selected_code]
close = df["Close"]
sma_s_series = sma(close, ma_short)
sma_l_series = sma(close, ma_long)
rsi_series = rsi(close, rsi_period)
vol_series = volatility(close, vol_window, ann_factor)

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**가격 + 이동평균**")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=close.index, y=close, name="종가", line=dict(width=2)))
    fig_price.add_trace(go.Scatter(x=close.index, y=sma_s_series, name=f"SMA{ma_short}"))
    fig_price.add_trace(go.Scatter(x=close.index, y=sma_l_series, name=f"SMA{ma_long}"))
    fig_price.update_layout(height=400, xaxis_title="날짜", yaxis_title="가격")
    st.plotly_chart(fig_price, use_container_width=True)

with col_b:
    st.markdown("**RSI**")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=rsi_series.index, y=rsi_series, name="RSI"))
    fig_rsi.add_hline(y=ob, line_dash="dash", line_color="red", annotation_text=f"과매수 {ob}")
    fig_rsi.add_hline(y=os_, line_dash="dash", line_color="green", annotation_text=f"과매도 {os_}")
    fig_rsi.update_layout(height=400, xaxis_title="날짜", yaxis_title="RSI", yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig_rsi, use_container_width=True)

st.markdown("**연환산 변동성**")
fig_vol = px.area(vol_series.dropna() * 100, labels={"value": "변동성 (%)", "Date": "날짜"})
fig_vol.update_layout(height=300, showlegend=False)
st.plotly_chart(fig_vol, use_container_width=True)

st.caption(
    "💡 Phase 1은 모니터링용입니다. 추천·비중 산출은 Phase 4(포트폴리오 최적화)에서 추가됩니다."
)
