from pathlib import Path

APP_TITLE = "Quantitative Asset Terminal"
APP_VERSION = "v2.6.0"

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

ASSETS = {
    "SPY": {
        "model_file": MODELS_DIR / "modelo_final.pkl",
        "margin_usd": 5.00,
        "start_date": "2000-01-01",
    },
    "BTC-USD": {
        "model_file": MODELS_DIR / "modelo_btc.pkl",
        "margin_usd": 0.00,
        "start_date": "2015-01-01",
    },
}

POPULAR_TICKERS = [
    "NVDA", "TSLA", "AAPL", "AMD", "MSFT", "AMZN", "GOOGL", "META",
    "SPY", "BTC-USD", "COIN", "PLTR", "MANUAL INPUT"
]

# Config de Plotly (zoom, export, etc.)
PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "toImageButtonOptions": {"format": "png", "scale": 2},
}
# config.py (añadir)
THEME = {
    "bg": "#0E1117",          # fondo Streamlit dark (aprox)
    "panel": "#111827",       # panel/plot background
    "grid": "rgba(255,255,255,0.08)",
    "text": "#E5E7EB",

    # tus acentos “terminal”
    "green": "#00FF99",
    "red": "#FF0055",
    "cyan": "#00D4FF",
    "yellow": "#FFD166",

    # series
    "price": "#00FF99",
    "range_fill": "rgba(0,255,153,0.18)",
    "limit": "#00FF99",
    "base_low": "#00D4FF",
    "sent_pos": "#00FF99",
    "sent_neg": "#FF0055",
    "sent_neu": "#AAAAAA",
}
