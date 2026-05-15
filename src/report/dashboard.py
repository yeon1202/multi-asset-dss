"""
다자산 의사결정 지원 시스템 — 통합 대시보드.

실행:  `streamlit run src/report/dashboard.py`

탭 구성:
  1) 📈 가격 모니터 (Phase 1)
     - 정규화 가격 차트 / RSI·이평·변동성 / 자산별 상세
  2) 💼 펀더멘털 스코어 (Phase 2)
     - DART 재무제표 + pykrx 시가총액으로 PER·PBR·ROE·OPM·부채비율
     - 0~100 점수 + 상위 N 종목

PROJECT_SPEC.md §2.2: 분석·추천만, 자동 주문 X.
"""
from __future__ import annotations

import os
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

from src.data.dart_loader import DartKeyMissingError, load_universe_financials
from src.data.market_cap_loader import load_market_caps
from src.data.price_loader import get_close_matrix, load_universe_prices
from src.indicators.fundamental import compute_ratios_table
from src.indicators.technical import rsi, sma, summarize, volatility
from src.scoring.fundamental_score import fundamental_score, top_n
from src.utils.config_loader import (
    load_fundamental_config,
    load_thresholds,
    load_universe,
)

# ----------------------------------------------------------------------
# 페이지 설정
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="다자산 의사결정 지원 시스템",
    page_icon="📊",
    layout="wide",
)

st.info(
    "ℹ️ 이 리포트는 **정보 제공 목적**이며, 투자 결정과 그 결과는 본인 책임입니다. "
    "자동 주문은 수행하지 않습니다."
)

st.title("📊 다자산 의사결정 지원 시스템")
st.caption("Phase 1 · 가격 모니터  |  Phase 2 · 펀더멘털 스코어링")

tab_price, tab_fundamental = st.tabs(["📈 가격 모니터", "💼 펀더멘털 스코어"])


# ======================================================================
# Tab 1 — Phase 1: 가격 모니터
# ======================================================================
@st.cache_data(ttl=60 * 60 * 6)
def _load_prices(lookback_days: int) -> dict[str, pd.DataFrame]:
    return load_universe_prices(lookback_days=lookback_days)


with tab_price:
    universe = load_universe()
    thresholds = load_thresholds()

    with st.sidebar:
        st.header("⚙️ 가격 모니터 설정")
        default_days = universe.get("default_lookback_days", 365)
        lookback = st.slider(
            "조회 기간 (일)", min_value=90, max_value=730,
            value=default_days, step=30, key="lookback",
        )
        st.caption(
            f"RSI: {thresholds['rsi']['period']}일 / "
            f"이평: {thresholds['moving_average']['short']}·"
            f"{thresholds['moving_average']['long']}일 / "
            f"변동성 윈도우: {thresholds['volatility']['window']}일"
        )

    with st.spinner("📡 가격 데이터 다운로드 중..."):
        prices = _load_prices(lookback)

    if not prices:
        st.error("데이터를 로드하지 못했습니다.")
        st.stop()

    name_by_code = {a["code"]: a["name"] for a in universe["assets"]}
    category_by_code = {a["code"]: a["category"] for a in universe["assets"]}

    # --- 정규화 가격 차트 ---
    st.subheader("1️⃣ 정규화 가격 (시작=100)")
    st.caption("같은 출발선에서 누가 더 잘 갔는지 한눈에 비교")

    closes = get_close_matrix(prices)
    normalized = closes.apply(lambda s: s / s.dropna().iloc[0] * 100)
    normalized = normalized.rename(columns=name_by_code)

    fig_norm = px.line(
        normalized,
        labels={"value": "정규화 가격", "Date": "날짜", "variable": "자산"},
    )
    fig_norm.update_layout(legend_title="자산", height=450)
    st.plotly_chart(fig_norm, use_container_width=True)

    # --- 지표 요약 표 ---
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
            df["Close"], rsi_period=rsi_period, ma_short=ma_short,
            ma_long=ma_long, vol_window=vol_window, annualize_factor=ann_factor,
        )
        last_close = s["last_close"]
        sma_l = s[f"sma_{ma_long}"]
        sma_s = s[f"sma_{ma_short}"]
        trend = "🟢 강세" if last_close > sma_l else "🔴 약세"
        r = s["rsi"]
        if pd.isna(r):
            rsi_sig = "—"
        elif r >= ob:
            rsi_sig = f"🔴 과매수 ({r:.1f})"
        elif r <= os_:
            rsi_sig = f"🟢 과매도 ({r:.1f})"
        else:
            rsi_sig = f"⚪ 중립 ({r:.1f})"
        rows.append({
            "종목": name_by_code.get(code, code),
            "카테고리": category_by_code.get(code, "-"),
            "종가": f"{last_close:,.0f}" if pd.notna(last_close) else "—",
            "RSI 신호": rsi_sig,
            f"SMA{ma_short}": f"{sma_s:,.0f}" if pd.notna(sma_s) else "—",
            f"SMA{ma_long}": f"{sma_l:,.0f}" if pd.notna(sma_l) else "—",
            "추세": trend,
            "연환산 변동성": f"{s['vol_annualized']*100:.1f}%" if pd.notna(s["vol_annualized"]) else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(f"RSI ≥ {ob}: 과매수 · RSI ≤ {os_}: 과매도 · 추세 = 종가 > SMA{ma_long} 강세")

    # --- 자산별 상세 ---
    st.subheader("3️⃣ 자산별 상세")
    selected_code = st.selectbox(
        "자산 선택", options=list(prices.keys()),
        format_func=lambda c: f"{name_by_code.get(c, c)} ({c})",
        key="selected_asset",
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
        fig_price.update_layout(height=400)
        st.plotly_chart(fig_price, use_container_width=True)
    with col_b:
        st.markdown("**RSI**")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=rsi_series.index, y=rsi_series, name="RSI"))
        fig_rsi.add_hline(y=ob, line_dash="dash", line_color="red", annotation_text=f"과매수 {ob}")
        fig_rsi.add_hline(y=os_, line_dash="dash", line_color="green", annotation_text=f"과매도 {os_}")
        fig_rsi.update_layout(height=400, yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_rsi, use_container_width=True)

    st.markdown("**연환산 변동성**")
    fig_vol = px.area(vol_series.dropna() * 100, labels={"value": "변동성 (%)"})
    fig_vol.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig_vol, use_container_width=True)


# ======================================================================
# Tab 2 — Phase 2: 펀더멘털 스코어
# ======================================================================
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)  # 24h
def _run_fundamental_pipeline(year: int) -> tuple[pd.DataFrame, dict]:
    """DART + pykrx 로 전 유니버스 비율·점수 계산. 캐시됨."""
    cfg = load_fundamental_config()
    codes = [a["code"] for a in cfg["universe"]]
    fin = load_universe_financials(codes, year=year, report_type=cfg["report_type"])
    mcap = load_market_caps(codes)
    ratios = compute_ratios_table(fin, mcap)
    scored = fundamental_score(
        ratios, weights=cfg["score_weights"], sanity_bounds=cfg.get("sanity_bounds"),
    )
    name_map = {a["code"]: a["name"] for a in cfg["universe"]}
    sector_map = {a["code"]: a["sector"] for a in cfg["universe"]}
    scored.insert(0, "name", scored.index.map(name_map))
    scored.insert(1, "sector", scored.index.map(sector_map))
    return scored, cfg


with tab_fundamental:
    cfg = load_fundamental_config()

    st.subheader("💼 펀더멘털 스코어")
    st.caption(
        f"KOSPI 시총 상위 {len(cfg['universe'])}종목 · "
        f"{cfg['target_year']}년 사업보고서 · "
        f"PER·PBR·ROE·영업이익률·부채비율 → 백분위 점수 합성"
    )

    # API 키 체크
    if not os.environ.get("DART_API_KEY", "").strip():
        st.warning(
            "⚠️ **DART_API_KEY 가 설정되지 않았습니다.**\n\n"
            "Phase 2 펀더멘털 스코어링은 DART API 키가 필요해요.\n\n"
            "1. https://opendart.fss.or.kr → '오픈API' → '인증키 신청'\n"
            "2. 발급받은 키를 프로젝트 루트의 `.env` 파일에 추가:\n"
            "   ```\n   DART_API_KEY=발급받은_40자리_키\n   ```\n"
            "3. 대시보드 새로고침"
        )
        st.stop()

    # 점수 가중치 표시
    with st.expander("📐 점수 가중치 (config/fundamental.yaml)"):
        weights_df = pd.DataFrame(
            list(cfg["score_weights"].items()), columns=["지표", "가중치"]
        )
        weights_df["방향"] = weights_df["지표"].map({
            "per": "낮을수록 좋음", "pbr": "낮을수록 좋음",
            "roe": "높을수록 좋음", "opm": "높을수록 좋음",
            "debt": "낮을수록 좋음",
        })
        st.dataframe(weights_df, hide_index=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        year = st.number_input(
            "회계연도", min_value=2015, max_value=2030,
            value=cfg["target_year"], step=1,
        )
        run_btn = st.button("🔄 분석 실행", type="primary")

    if run_btn or st.session_state.get("fund_scored") is not None:
        if run_btn:
            try:
                with st.spinner(f"📡 DART {year}년 재무제표 {len(cfg['universe'])}종목 다운로드 중... (수 분 소요)"):
                    scored, _ = _run_fundamental_pipeline(year)
                st.session_state["fund_scored"] = scored
            except DartKeyMissingError as e:
                st.error(str(e))
                st.stop()
            except Exception as e:  # noqa: BLE001
                st.error(f"분석 실패: {e}")
                st.stop()

        scored: pd.DataFrame = st.session_state["fund_scored"]

        # --- 상위 N ---
        st.markdown(f"### 🏆 종합 점수 상위 {cfg['top_n']}")
        leaders = top_n(scored, n=cfg["top_n"])
        display_cols = [
            "name", "sector", "composite_score",
            "per", "pbr", "roe", "opm", "debt",
        ]
        leaders_view = leaders[display_cols].copy()
        leaders_view["composite_score"] = leaders_view["composite_score"].round(1)
        for c in ["per", "pbr", "debt"]:
            leaders_view[c] = leaders_view[c].round(2)
        for c in ["roe", "opm"]:
            leaders_view[c] = (leaders_view[c] * 100).round(1).astype(str) + "%"
        leaders_view.columns = ["종목", "섹터", "종합점수", "PER", "PBR", "ROE", "영업이익률", "부채비율"]
        st.dataframe(leaders_view, use_container_width=True)
        st.caption(
            "ℹ️ 점수는 **유니버스 내 백분위**입니다. 절대 가치 평가가 아니라 "
            "현재 30종목 안에서의 상대 순위입니다."
        )

        # --- 점수 분포 차트 ---
        st.markdown("### 📊 종합 점수 분포")
        chart_df = scored.dropna(subset=["composite_score"]).copy()
        chart_df = chart_df.sort_values("composite_score", ascending=True)
        fig_score = px.bar(
            chart_df, x="composite_score", y="name", color="sector",
            orientation="h", height=600,
            labels={"composite_score": "종합 점수 (0-100)", "name": "종목"},
        )
        st.plotly_chart(fig_score, use_container_width=True)

        # --- 전체 점수표 ---
        with st.expander(f"📋 전체 {len(scored)}종목 상세 ({year}년 기준)"):
            full = scored.copy()
            full["composite_score"] = full["composite_score"].round(1)
            st.dataframe(full, use_container_width=True)

        # --- CSV 다운로드 ---
        csv_bytes = scored.to_csv(encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "💾 결과 CSV 다운로드", data=csv_bytes,
            file_name=f"fundamental_{year}.csv", mime="text/csv",
        )
    else:
        st.info("👈 좌측의 '🔄 분석 실행' 버튼을 눌러주세요. 첫 실행 시 수 분 소요됩니다 (이후 24시간 캐시).")
