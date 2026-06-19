from pathlib import Path
import joblib
import streamlit as st

@st.cache_resource
def load_brain(model_path: str | Path):
    """Carga el archivo .pkl (SPY o BTC)"""
    model_path = Path(model_path)
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        return None
