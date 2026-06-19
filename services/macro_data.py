import datetime
import pandas as pd
import streamlit as st
import pandas_datareader.data as web

@st.cache_data(ttl=3600 * 12)
def get_fred_data():
    start = datetime.datetime(2018, 1, 1)
    end = datetime.datetime.now()

    indicators = {
        "M2 Money Supply": "M2SL",
        "Fed Total Assets": "WALCL",
        "HY Option-Adjusted Spread": "BAMLH0A0HYM2",
        "10Y-2Y Yield Curve": "T10Y2Y",
    }

    data: dict[str, pd.DataFrame] = {}
    for name, code in indicators.items():
        try:
            df = web.DataReader(code, "fred", start, end)
            data[name] = df
        except Exception:
            pass
    return data
