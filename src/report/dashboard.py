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
from datetime import date
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
from src.data.macro_loader import (
    EcosKeyMissingError,
    FredKeyMissingError,
    load_all_macro,
)
from src.data.market_cap_loader import load_market_caps
from src.data.price_loader import get_close_matrix, load_universe_prices
from src.indicators.fundamental import compute_ratios_table
from src.indicators.technical import rsi, sma, summarize, volatility
from src.portfolio.kelly import apply_kelly_sizing
from src.portfolio.optimizer import (
    equal_weight_fallback,
    optimize_max_sharpe,
)
from src.regime.detector import classify_regime, detect_history
from src.scoring.composite_score import composite_score
from src.scoring.fundamental_score import fundamental_score, top_n
from src.scoring.regime_fit import asset_fit_table
from src.utils.config_loader import (
    load_fundamental_config,
    load_macro_config,
    load_portfolio_config,
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
st.caption(
    "Phase 1 · 가격 모니터  |  Phase 2 · 펀더멘털  |  "
    "Phase 3 · 레짐  |  Phase 4 · 포트폴리오"
)

tab_price, tab_fundamental, tab_regime, tab_portfolio = st.tabs(
    ["📈 가격 모니터", "💼 펀더멘털 스코어", "🌐 시장 국면", "🎯 포트폴리오"]
)


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


# ======================================================================
# Tab 3 — Phase 3: 시장 국면(레짐)
# ======================================================================
@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def _run_macro_pipeline() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """ECOS + FRED 거시 데이터 + 레짐 히스토리. 6h 캐시."""
    cfg = load_macro_config()
    macro = load_all_macro()
    history = detect_history(macro, cfg["regime"])
    return macro, history, cfg


with tab_regime:
    st.subheader("🌐 시장 국면(레짐) 판단")
    st.caption("거시 지표 → Risk-On / Risk-Off / Neutral 룰 기반 분류. 자산별 적합도 추가.")

    # 키 체크
    missing_keys = []
    if not os.environ.get("ECOS_API_KEY", "").strip():
        missing_keys.append("ECOS_API_KEY (한국은행)")
    if not os.environ.get("FRED_API_KEY", "").strip():
        missing_keys.append("FRED_API_KEY (St. Louis Fed)")

    if missing_keys:
        st.warning(
            f"⚠️ 다음 API 키가 .env 에 없습니다: **{', '.join(missing_keys)}**\n\n"
            "발급 (둘 다 무료):\n"
            "- ECOS: https://ecos.bok.or.kr/api → '인증키 신청'\n"
            "- FRED: https://fred.stlouisfed.org/docs/api/api_key.html\n\n"
            "발급 후 `.env` 에 추가하고 새로고침."
        )
        st.stop()

    if st.button("🔄 거시 데이터 갱신", type="primary", key="refresh_regime"):
        st.cache_data.clear()

    try:
        with st.spinner("📡 ECOS · FRED 데이터 로드..."):
            macro, history, mcfg = _run_macro_pipeline()
    except (EcosKeyMissingError, FredKeyMissingError) as e:
        st.error(str(e))
        st.stop()
    except Exception as e:  # noqa: BLE001
        st.error(f"거시 데이터 로드 실패: {e}")
        st.stop()

    # 가장 최근 시점
    latest = macro.iloc[-1]
    latest_date = latest.name
    values = {n: latest.get(n, float("nan")) for n in mcfg["regime"]["features"]}
    result = classify_regime(values, mcfg["regime"])

    # --- 메인 표시 ---
    color_map = {"risk_on": "🟢", "risk_off": "🔴", "neutral": "⚪"}
    label_kr = {"risk_on": "RISK-ON (위험선호)",
                "risk_off": "RISK-OFF (위험회피)",
                "neutral": "NEUTRAL (중립)"}

    col1, col2, col3 = st.columns([1, 1, 1])
    col1.metric(
        "현재 레짐",
        f"{color_map.get(result.label, '?')} {label_kr.get(result.label, result.label)}",
    )
    col2.metric(
        "종합 점수",
        f"{result.score:+.3f}",
        help="-1=극도의 risk-off, +1=극도의 risk-on",
    )
    col3.metric("기준일", str(latest_date.date()))

    # --- feature 기여도 ---
    st.markdown("### 📐 지표별 기여도 (현재)")
    contrib_rows = []
    feat_labels = {
        "vix": "VIX",
        "hy_oas": "하이일드 OAS",
        "spread_10_2": "10Y-2Y 스프레드",
        "usd_krw": "원/달러 환율",
        "base_rate": "한국 기준금리",
    }
    for feat in mcfg["regime"]["features"]:
        contrib_rows.append({
            "지표": feat_labels.get(feat, feat),
            "현재값": f"{result.feature_values.get(feat, float('nan')):.3f}"
                     if pd.notna(result.feature_values.get(feat, float('nan'))) else "—",
            "정규화점수": f"{result.feature_scores.get(feat, float('nan')):+.2f}"
                          if pd.notna(result.feature_scores.get(feat, float('nan'))) else "—",
            "기여도": f"{result.contributions.get(feat, 0):+.3f}",
            "가중치": f"{mcfg['regime']['features'][feat]['weight']:.2f}",
        })
    st.dataframe(pd.DataFrame(contrib_rows), use_container_width=True, hide_index=True)

    # --- 종합 점수 히스토리 ---
    st.markdown("### 📈 레짐 점수 히스토리")
    valid_hist = history.dropna(subset=["score"])
    if not valid_hist.empty:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=valid_hist.index, y=valid_hist["score"],
            name="종합 점수", line=dict(width=2),
            fill="tozeroy",
        ))
        fig_hist.add_hline(y=mcfg["regime"]["thresholds"]["risk_on"],
                          line_dash="dash", line_color="green",
                          annotation_text="Risk-On 임계")
        fig_hist.add_hline(y=mcfg["regime"]["thresholds"]["risk_off"],
                          line_dash="dash", line_color="red",
                          annotation_text="Risk-Off 임계")
        fig_hist.add_hline(y=0, line_color="gray", line_width=1)
        fig_hist.update_layout(
            height=400, yaxis=dict(range=[-1.1, 1.1], title="점수"),
            xaxis_title="날짜",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # --- 자산 적합도 ---
    st.markdown("### 🎯 자산별 레짐 적합도")
    st.caption("현재 레짐에서 각 자산이 얼마나 적합한지 0~100. 50=중립.")
    fit = asset_fit_table(mcfg["asset_preference"], result.score)
    fit = fit.sort_values("regime_fit", ascending=False)

    # 자산명 매핑 (Phase 1 유니버스)
    asset_names = {a["code"]: a["name"] for a in load_universe()["assets"]}
    fit_view = fit.copy()
    fit_view.insert(0, "name", fit_view.index.map(asset_names))
    fit_view["preference"] = fit_view["preference"].round(2)
    fit_view["regime_fit"] = fit_view["regime_fit"].round(1)
    fit_view.columns = ["종목", "선호도 (-1~+1)", "적합도 (0~100)"]
    st.dataframe(fit_view, use_container_width=True)

    fig_fit = px.bar(
        fit_view.reset_index().rename(columns={"index": "코드"}),
        x="적합도 (0~100)", y="종목",
        orientation="h", height=350,
        color="적합도 (0~100)", color_continuous_scale="RdYlGn",
        range_color=[0, 100],
    )
    fig_fit.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_fit, use_container_width=True)

    # --- 거시 지표 차트 ---
    with st.expander("📊 거시 지표 원본 차트"):
        feature_to_plot = st.selectbox(
            "지표 선택",
            options=list(mcfg["regime"]["features"].keys()),
            format_func=lambda f: feat_labels.get(f, f),
        )
        if feature_to_plot in macro.columns:
            series = macro[feature_to_plot].dropna()
            fig_m = px.line(series, labels={"value": feat_labels.get(feature_to_plot, feature_to_plot)})
            fig_m.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_m, use_container_width=True)

    st.caption(
        "ℹ️ Phase 3 의 적합도 점수는 **레짐 한 가지** 만 반영합니다. "
        "Phase 4 (포트폴리오 최적화) 에서 기술적·펀더멘털·레짐을 통합한 비중을 산출할 예정입니다."
    )


# ======================================================================
# Tab 4 — Phase 4: 포트폴리오 최적화
# ======================================================================
@st.cache_data(ttl=60 * 60 * 3, show_spinner=False)
def _run_portfolio_pipeline() -> dict:
    """가격 + 거시 + 점수 + 최적화 + 켈리. 3시간 캐시."""
    macro_cfg = load_macro_config()
    port_cfg = load_portfolio_config()
    universe = load_universe()

    # 가격
    prices = load_universe_prices()
    closes = get_close_matrix(prices)

    # 레짐
    macro = load_all_macro()
    latest = macro.iloc[-1]
    values = {n: latest.get(n, float("nan")) for n in macro_cfg["regime"]["features"]}
    regime_res = classify_regime(values, macro_cfg["regime"])

    # 통합 점수
    scores = composite_score(
        prices=closes,
        regime_score=regime_res.score,
        preferences=macro_cfg["asset_preference"],
        weights=port_cfg["score_weights"],
        technical_cfg=port_cfg["technical_score"],
    )

    # 최적화
    constraints = port_cfg["constraints"]
    try:
        opt = optimize_max_sharpe(
            prices=closes,
            composite_score=scores["composite"],
            max_expected_return=port_cfg["expected_return_mapping"]["max_expected_return"],
            max_weight_per_asset=constraints["max_weight_per_asset"],
            min_weight_per_asset=constraints["min_weight_per_asset"],
            risk_free_rate=constraints["risk_free_rate"],
        )
        risky = opt.weights
        port_ret, port_vol, port_sharpe = opt.expected_return, opt.volatility, opt.sharpe
        fallback_used = False
    except Exception as e:  # noqa: BLE001
        risky = equal_weight_fallback(scores["composite"])
        daily = closes.pct_change().dropna()
        port_ret = float((risky * daily.mean()).sum() * 252)
        port_vol = float((daily @ risky.values).std() * (252 ** 0.5))
        port_sharpe = (
            (port_ret - constraints["risk_free_rate"]) / port_vol if port_vol > 0 else 0.0
        )
        fallback_used = True

    # Kelly
    kelly_cfg = port_cfg["kelly"]
    final = apply_kelly_sizing(
        risky_weights=risky,
        expected_return=port_ret,
        volatility=port_vol,
        risk_free_rate=constraints["risk_free_rate"],
        kelly_fraction_param=kelly_cfg["fraction"],
        max_total_risk_weight=kelly_cfg["max_total_risk_weight"],
        min_cash_weight=constraints["min_cash_weight"],
    )

    return {
        "scores": scores,
        "risky_weights": risky,
        "final": final,
        "port_ret": port_ret,
        "port_vol": port_vol,
        "port_sharpe": port_sharpe,
        "regime_label": regime_res.label,
        "regime_score": regime_res.score,
        "fallback_used": fallback_used,
        "universe": universe,
    }


with tab_portfolio:
    st.subheader("🎯 통합 포트폴리오 추천")
    st.caption(
        "Phase 1 모멘텀 + Phase 3 레짐 → 통합 점수 → 마코위츠 최적화 → Half-Kelly. "
        "현금 비중 자동 산출."
    )

    if not os.environ.get("ECOS_API_KEY", "").strip() or not os.environ.get("FRED_API_KEY", "").strip():
        st.warning(
            "⚠️ Phase 4 는 Phase 3 거시 데이터(레짐)를 필요로 합니다. "
            "ECOS_API_KEY · FRED_API_KEY 를 `.env` 에 먼저 설정해주세요."
        )
        st.stop()

    if st.button("🔄 포트폴리오 재계산", type="primary", key="refresh_portfolio"):
        st.cache_data.clear()

    try:
        with st.spinner("📊 가격 + 거시 + 최적화 실행 중..."):
            pkg = _run_portfolio_pipeline()
    except Exception as e:  # noqa: BLE001
        st.error(f"파이프라인 실패: {e}")
        st.stop()

    if pkg["fallback_used"]:
        st.warning("⚠️ 마코위츠 최적화 실패 — 동일 가중 fallback 사용 중.")

    asset_names = {a["code"]: a["name"] for a in pkg["universe"]["assets"]}

    # --- 통합 점수표 ---
    st.markdown("### 🎯 자산별 점수")
    scores = pkg["scores"].copy()
    scores.insert(0, "종목", scores.index.map(asset_names))
    scores = scores.rename(columns={
        "technical": "기술적 (모멘텀)",
        "regime": "레짐 적합도",
        "composite": "종합 점수",
    })
    scores_disp = scores.copy()
    for c in ("기술적 (모멘텀)", "레짐 적합도", "종합 점수"):
        scores_disp[c] = scores_disp[c].round(1)
    st.dataframe(scores_disp, use_container_width=True)

    # --- 추천 비중 ---
    st.markdown("### 💼 추천 포트폴리오")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("기대수익률 (연)", f"{pkg['port_ret']*100:+.2f}%")
    col_b.metric("변동성 (연)", f"{pkg['port_vol']*100:.2f}%")
    col_c.metric("샤프 비율", f"{pkg['port_sharpe']:.2f}")
    col_d.metric(
        "위험자산 비중",
        f"{(1 - pkg['final'].get('CASH', 0.0)) * 100:.1f}%",
    )

    # 파이 차트
    final = pkg["final"]
    pie_df = pd.DataFrame({
        "자산": [
            "💵 현금" if c == "CASH" else asset_names.get(c, c)
            for c in final.index
        ],
        "비중": (final.values * 100).round(2),
    })
    pie_df = pie_df[pie_df["비중"] > 0.05]
    fig_pie = px.pie(pie_df, names="자산", values="비중", hole=0.35)
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(height=450)
    st.plotly_chart(fig_pie, use_container_width=True)

    # 표
    alloc_view = pd.DataFrame({
        "자산": pie_df["자산"],
        "비중 (%)": pie_df["비중"],
    }).sort_values("비중 (%)", ascending=False)
    st.dataframe(alloc_view, use_container_width=True, hide_index=True)

    st.caption(
        "💡 Half-Kelly 적용. 점수가 같아도 변동성이 큰 자산은 비중이 낮아집니다. "
        "이 추천은 **백테스팅 미반영** 휴리스틱 — Phase 5 에서 검증 예정입니다."
    )
