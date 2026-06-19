import warnings
import streamlit as st

from config import APP_VERSION, ASSETS, POPULAR_TICKERS, PLOTLY_CONFIG, THEME
from services.models import load_brain
from services.features import get_features_complete
from services.macro_data import get_fred_data
from services.news_sentiment import ensure_nltk, get_finviz_data_robust, get_keywords
from services.market_data import get_intraday_price
from charts.plotly_charts import predictor_chart, fred_line, sentiment_price_chart, keywords_bar


warnings.filterwarnings("ignore")

st.set_page_config(page_title="Quantitative Asset Terminal", layout="wide", page_icon="📊")

ensure_nltk()

st.markdown("""
<style>
.stButton>button {
    width: 100%;
    font-family: 'Segoe UI', sans-serif;
    font-size: 16px;
    font-weight: 600;
    background-color: #0E1117;
    color: #00FF99;
    border: 1px solid #00FF99;
    border-radius: 4px;
    padding: 12px;
    transition: all 0.3s;
}
.stButton>button:hover {
    background-color: #00FF99;
    color: #000000;
}
.metric-container {background-color: #1E1E1E; padding: 15px; border-radius: 5px; border-left: 3px solid #00FF99;}
</style>
""", unsafe_allow_html=True)

st.title("QUANTITATIVE ASSET TERMINAL")
st.markdown(f"**INSTITUTIONAL GRADE ANALYTICS** | `{APP_VERSION}` | 🟢 SYSTEM OPERATIONAL")

tab1, tab2, tab3 = st.tabs(["📉 ALGORITHMIC PREDICTOR", "🏦 MACRO LIQUIDITY MONITOR", "📰 SENTIMENT & NLP ENGINE"])

# =========================
# TAB 1: PREDICTOR
# =========================
with tab1:
    col_t, col_btn = st.columns([1, 3])
    with col_t:
        asset_choice = st.selectbox("SELECT ASSET", list(ASSETS.keys()))

    cfg = ASSETS[asset_choice]
    target_file = cfg["model_file"]
    margin_usd = float(cfg["margin_usd"])

    st.markdown(f"### RISK MODEL: {asset_choice}")
    st.caption(f"Architecture: XGBoost Hybrid | Loading Weights: `{target_file.name}` | Safety Margin: ${margin_usd:,.2f}")

    if st.button("EXECUTE PREDICTION MODEL", key="btn_pred"):
        brain = load_brain(target_file)

        if brain is None:
            st.error(f"❌ ERROR: No se encuentra el archivo '{target_file}'.")
        else:
            with st.spinner("Processing Market Data & Calculating Inference..."):
                try:
                    windows, df_daily = get_features_complete(asset_choice, brain["features"], n_months=3)

                    if windows is None:
                        st.error("Insufficient market data.")
                    else:
                        month_levels = []
                        month_colors = [THEME["limit"], THEME["cyan"], THEME["yellow"]]

                        for j, w in enumerate(windows):
                            X_input = w["x_input"]
                            prev_close = w["prev_close"]

                            try:
                                p_xgb = brain["xgb_model"].predict(
                                    X_input, iteration_range=(0, brain["best_iter"] + 1)
                                )
                            except Exception:
                                p_xgb = brain["xgb_model"].predict(X_input)

                            p_ridge = brain["ridge_model"].predict(X_input)
                            pred_drop = 0.7 * p_xgb + 0.3 * p_ridge

                            base_low = float(prev_close * (1 + pred_drop) + brain["bias"])
                            limit_order = base_low + margin_usd

                            month_levels.append({
                                "label": w["label"],
                                "month_start": w["month_start"],
                                "month_end": w["month_end"],
                                "base_low": base_low,
                                "limit_order": limit_order,
                                "color": month_colors[min(j, len(month_colors) - 1)],
                            })

                        # Métricas: mes calendario actual (primer elemento)
                        curr = month_levels[0]
                        curr_prev_close = windows[0]["prev_close"]
                        curr_price = float(df_daily["Close"].iloc[-1])

                        c1, c2, c3 = st.columns(3)
                        c1.metric("PREV MONTH CLOSE", f"${curr_prev_close:,.2f}")
                        c2.metric(
                            "LIMIT BUY ORDER (AI)",
                            f"${curr['limit_order']:,.2f}",
                            delta=f"Base Low: ${curr['base_low']:,.2f}",
                            delta_color="off",
                        )
                        c3.metric(
                            "MARKET PRICE (LIVE)",
                            f"${curr_price:,.2f}",
                            delta=f"{curr_price - curr['limit_order']:,.2f} spread",
                            delta_color="inverse",
                        )

                        st.divider()

                        fig = predictor_chart(
                            df_daily=df_daily,
                            title=f"{asset_choice} — Price / Range / AI Levels (last 90 sessions)",
                            month_levels=month_levels,
                        )
                        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

                except Exception as e:
                    st.error(f"RUNTIME ERROR: {str(e)}")

# =========================
# TAB 2: MACRO (SIN CAMBIOS de API)
# =========================
with tab2:
    st.markdown("### FEDERAL RESERVE & LIQUIDITY DASHBOARD")

    if st.button("FETCH FRED DATA", key="btn_macro"):
        with st.spinner("Establishing Connection to St. Louis Fed API..."):
            macro = get_fred_data()

            c1, c2 = st.columns(2)
            with c1:
                if "M2 Money Supply" in macro:
                    st.plotly_chart(
                        fred_line(macro["M2 Money Supply"], "M2 Money Supply"),
                        use_container_width=True,
                        config=PLOTLY_CONFIG
                    )

                if "HY Option-Adjusted Spread" in macro:
                    st.plotly_chart(
                        fred_line(macro["HY Option-Adjusted Spread"], "High Yield Credit Spread", hline=5.0),
                        use_container_width=True,
                        config=PLOTLY_CONFIG
                    )

            with c2:
                if "Fed Total Assets" in macro:
                    st.plotly_chart(
                        fred_line(macro["Fed Total Assets"], "Fed Balance Sheet"),
                        use_container_width=True,
                        config=PLOTLY_CONFIG
                    )

                if "10Y-2Y Yield Curve" in macro:
                    st.plotly_chart(
                        fred_line(macro["10Y-2Y Yield Curve"], "Yield Curve (10Y-2Y)", hline=0.0),
                        use_container_width=True,
                        config=PLOTLY_CONFIG
                    )

# =========================
# TAB 3: SENTIMENT + NLP (SIN CAMBIOS de API)
# =========================
with tab3:
    st.markdown("### NLP SENTIMENT ENGINE & NEWS FLOW")

    col_sel1, col_sel2 = st.columns([1, 2])
    with col_sel1:
        ticker_select = st.selectbox("ASSET SELECTION", POPULAR_TICKERS)
    with col_sel2:
        target_ticker = st.text_input("ENTER TICKER SYMBOL:", "").upper() if ticker_select == "MANUAL INPUT" else ticker_select

    if st.button(f"ANALYZE: {target_ticker}", key="btn_news") and target_ticker:
        with st.spinner(f"Scraping Finviz & Processing NLP for {target_ticker}..."):
            df_news = get_finviz_data_robust(target_ticker)

            if df_news.empty:
                st.warning(f"No sufficient data found for {target_ticker}.")
            else:
                min_date, max_date = df_news["datetime"].min(), df_news["datetime"].max()
                df_price = get_intraday_price(target_ticker, min_date, max_date)
                keywords = get_keywords(df_news, target_ticker)

                avg_sent = float(df_news["sentiment"].mean())
                sent_color = "#00FF99" if avg_sent > 0.05 else "#FF0055" if avg_sent < -0.05 else "#AAAAAA"

                m1, m2, m3 = st.columns(3)
                m1.metric("NEWS VOLUME", len(df_news))
                m2.markdown(
                    f"<div style='text-align:center'><b>SENTIMENT SCORE</b><br>"
                    f"<span style='color:{sent_color}; font-size:28px; font-weight:bold'>{avg_sent:.4f}</span></div>",
                    unsafe_allow_html=True
                )
                m3.metric("TIMEFRAME", f"{min_date.strftime('%m/%d')} - {max_date.strftime('%m/%d')}")

                st.divider()

                fig_main = sentiment_price_chart(
                    df_price=df_price,
                    df_news=df_news,
                    title=f"PRICE vs SENTIMENT: {target_ticker}"
                )
                st.plotly_chart(fig_main, use_container_width=True, config=PLOTLY_CONFIG)

                fig_kw = keywords_bar(keywords, title="TOP KEYWORDS")
                st.plotly_chart(fig_kw, use_container_width=True, config=PLOTLY_CONFIG)

                with st.expander("VIEW RAW NEWS FEED"):
                    st.dataframe(df_news[["datetime", "headline", "sentiment"]], use_container_width=True)
