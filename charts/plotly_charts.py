import numpy as np
import pandas as pd
import plotly.graph_objects as go
from config import THEME


def predictor_chart(
    df_daily: pd.DataFrame,
    title: str,
    limit_order: float | None = None,
    base_low: float | None = None,
    month_levels: list[dict] | None = None,
):
    df_viz = df_daily.iloc[-90:].copy()
    fig = go.Figure()

    # Banda High-Low (relleno verde suave)
    fig.add_trace(go.Scatter(
        x=df_viz.index, y=df_viz["High"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
        name="High"
    ))
    fig.add_trace(go.Scatter(
        x=df_viz.index, y=df_viz["Low"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor=THEME["range_fill"],
        name="Daily Range",
        hoverinfo="skip",
    ))

    # Close
    fig.add_trace(go.Scatter(
        x=df_viz.index, y=df_viz["Close"],
        mode="lines",
        name="Market Price",
        line=dict(color=THEME["price"], width=2),
    ))

    # ===== NUEVO: niveles por mes (segmentos) =====
    if month_levels:
        for lvl in month_levels:
            fig.add_trace(go.Scatter(
                x=[lvl["month_start"], lvl["month_end"]],
                y=[lvl["limit_order"], lvl["limit_order"]],
                mode="lines",
                name=f"LIMIT {lvl['label']}",
                line=dict(color=lvl["color"], width=3),
            ))
            fig.add_trace(go.Scatter(
                x=[lvl["month_start"], lvl["month_end"]],
                y=[lvl["base_low"], lvl["base_low"]],
                mode="lines",
                name=f"BASE {lvl['label']}",
                line=dict(color=lvl["color"], width=2, dash="dot"),
                opacity=0.9,
            ))

    # ===== Compat: modo antiguo (hlines) =====
    elif (limit_order is not None) and (base_low is not None):
        fig.add_hline(
            y=float(limit_order),
            line_width=3,
            line_dash="solid",
            line_color=THEME["limit"],
            annotation_text="ALGO BUY LIMIT",
            annotation_position="top left",
            annotation_font_color=THEME["limit"],
        )
        fig.add_hline(
            y=float(base_low),
            line_width=2,
            line_dash="dot",
            line_color=THEME["base_low"],
            annotation_text="AI PROJECTED LOW",
            annotation_position="top left",
            annotation_font_color=THEME["base_low"],
        )

    _apply_terminal_layout(fig, title=title, height=520)
    return fig


def sentiment_price_chart(
    df_price: pd.DataFrame,
    df_news: pd.DataFrame,
    title: str,
    neutral_band: float = 0.05,
    show_neutral: bool = False,
    add_sentiment_ma: bool = True,
    ma_window: int = 7,
):
    fig = go.Figure()

    # Precio (izquierda)
    if df_price is not None and not df_price.empty and "Close" in df_price.columns:
        price_series = df_price["Close"].copy()

        fig.add_trace(go.Scatter(
            x=price_series.index,
            y=price_series.values,
            mode="lines",
            name="Price Action",
            line=dict(color=THEME["green"], width=2),
            opacity=0.55,
            hovertemplate="Date: %{x}<br>Close: %{y:.2f}<extra></extra>",
            yaxis="y"
        ))

    # Sentiment (derecha)
    dfn = df_news.copy()
    dfn = dfn.dropna(subset=["datetime", "sentiment", "headline"])
    dfn = dfn.sort_values("datetime")

    s = dfn["sentiment"].astype(float).values

    def cls(v: float) -> str:
        if v > neutral_band:
            return "pos"
        if v < -neutral_band:
            return "neg"
        return "neu"

    classes = np.array([cls(v) for v in s])
    sizes = (np.abs(s) * 14 + 6).clip(6, 22)

    def add_scatter(mask, name, color, opacity):
        sub = dfn.loc[mask]
        if sub.empty:
            return
        fig.add_trace(go.Scatter(
            x=sub["datetime"],
            y=sub["sentiment"],
            mode="markers",
            name=name,
            marker=dict(
                size=sizes[mask],
                color=color,
                opacity=opacity,
                line=dict(width=1, color="rgba(255,255,255,0.25)"),
            ),
            hovertemplate=(
                "Date: %{x}<br>"
                "Sentiment: %{y:.3f}<br>"
                "<br><b>%{customdata}</b>"
                "<extra></extra>"
            ),
            customdata=sub["headline"].values,
            yaxis="y2"
        ))

    pos_mask = classes == "pos"
    neg_mask = classes == "neg"
    neu_mask = classes == "neu"

    add_scatter(pos_mask, "Sentiment +", THEME["green"], 0.95)
    add_scatter(neg_mask, "Sentiment -", THEME["red"], 0.95)

    if show_neutral:
        add_scatter(neu_mask, "Sentiment 0", THEME["sent_neu"], 0.45)

    fig.add_hline(
        y=0,
        line_width=1,
        line_dash="dash",
        line_color="rgba(255,255,255,0.35)",
        annotation_text="0",
        annotation_position="top left",
        annotation_font_color="rgba(255,255,255,0.55)",
    )

    fig.add_hrect(
        y0=-neutral_band,
        y1=neutral_band,
        fillcolor="rgba(255,255,255,0.04)",
        line_width=0,
        layer="below",
    )

    if add_sentiment_ma and len(dfn) >= ma_window:
        ma = dfn.set_index("datetime")["sentiment"].rolling(ma_window).mean()
        fig.add_trace(go.Scatter(
            x=ma.index,
            y=ma.values,
            mode="lines",
            name=f"Sentiment MA({ma_window})",
            line=dict(color=THEME["cyan"], width=2),
            opacity=0.7,
            yaxis="y2",
            hovertemplate="Date: %{x}<br>MA: %{y:.3f}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        height=600,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor=THEME["bg"],
        plot_bgcolor=THEME["bg"],
        font=dict(color=THEME["text"]),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0
        ),
        yaxis=dict(
            title="Price",
            gridcolor=THEME["grid"],
            zerolinecolor=THEME["grid"],
            tickfont=dict(color=THEME["text"]),
        ),
        yaxis2=dict(
            title="Sentiment",
            overlaying="y",
            side="right",
            range=[-1, 1],
            gridcolor="rgba(0,0,0,0)",
            tickfont=dict(color=THEME["text"]),
            zeroline=False,
        ),
    )

    fig.update_xaxes(
        rangeslider_visible=True,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        gridcolor=THEME["grid"],
        tickfont=dict(color=THEME["text"]),
    )

    return fig


def keywords_bar(keywords: list[tuple[str, int]], title="TOP KEYWORDS"):
    fig = go.Figure()

    if not keywords:
        _apply_terminal_layout(fig, title=title, height=280)
        return fig

    words, counts = zip(*keywords)

    fig.add_trace(go.Bar(
        x=list(counts),
        y=[w.upper() for w in words],
        orientation="h",
        marker=dict(color=THEME["green"]),
        opacity=0.75,
        name="Count",
    ))

    fig.update_yaxes(autorange="reversed")
    _apply_terminal_layout(fig, title=title, height=300)
    return fig


def fred_line(df: pd.DataFrame, title: str, hline: float | None = None):
    series = df.iloc[:, 0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=title))
    if hline is not None:
        fig.add_hline(y=float(hline), line_width=1, line_dash="dash", opacity=0.7)
    fig.update_layout(
        title=title.upper(),
        template="plotly_dark",
        height=320,
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified",
    )
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def _apply_terminal_layout(fig, title: str, height: int = 520):
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor=THEME["bg"],
        plot_bgcolor=THEME["bg"],
        font=dict(color=THEME["text"]),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0,
            font=dict(color=THEME["text"])
        ),
    )
    fig.update_xaxes(
        rangeslider_visible=True,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        gridcolor=THEME["grid"],
        zerolinecolor=THEME["grid"],
        tickfont=dict(color=THEME["text"]),
    )
    fig.update_yaxes(
        showspikes=True,
        gridcolor=THEME["grid"],
        zerolinecolor=THEME["grid"],
        tickfont=dict(color=THEME["text"]),
    )
    return fig
