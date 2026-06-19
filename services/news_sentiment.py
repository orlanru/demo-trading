import datetime
import pandas as pd
import streamlit as st
import nltk

from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from collections import Counter

def ensure_nltk():
    # Descargas NLTK (como en tu código)
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

    try:
        nltk.data.find("corpora/stopwords.zip")
    except LookupError:
        nltk.download("stopwords", quiet=True)

def parse_finviz_date(date_str, time_str, last_date):
    today = datetime.datetime.now()
    if date_str is None:
        return datetime.datetime.combine(last_date, datetime.datetime.strptime(time_str, "%I:%M%p").time())

    date_str = date_str.replace("Today", today.strftime("%b-%d"))
    try:
        current_dt = datetime.datetime.strptime(f"{date_str} {time_str}", "%b-%d-%y %I:%M%p")
    except ValueError:
        temp_dt = datetime.datetime.strptime(f"{date_str} {time_str}", "%b-%d %I:%M%p")
        current_dt = temp_dt.replace(year=today.year)
        if current_dt > today + datetime.timedelta(days=1):
            current_dt = current_dt.replace(year=today.year - 1)
    return current_dt

@st.cache_data(ttl=600)
def get_finviz_data_robust(ticker: str) -> pd.DataFrame:
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    req = Request(url=url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        response = urlopen(req)
        html = BeautifulSoup(response, "html.parser")
        news_table = html.find(id="news-table")
        if not news_table:
            return pd.DataFrame()

        parsed_data = []
        last_date = datetime.datetime.now().date()

        for tr in news_table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 2:
                continue

            text = tds[1].a.get_text()
            date_info = tds[0].text.strip().split()

            if len(date_info) == 1:
                dt_obj = parse_finviz_date(None, date_info[0], last_date)
            else:
                dt_obj = parse_finviz_date(date_info[0], date_info[1], last_date)
                last_date = dt_obj.date()

            parsed_data.append([dt_obj, text])

        df = pd.DataFrame(parsed_data, columns=["datetime", "headline"])

        vader = SentimentIntensityAnalyzer()
        df["sentiment"] = df["headline"].apply(lambda x: vader.polarity_scores(x)["compound"])
        return df.sort_values("datetime")

    except Exception:
        return pd.DataFrame()

def get_keywords(df: pd.DataFrame, ticker: str):
    stop_words = set(stopwords.words("english"))
    ignore = {
        "stock","stocks","market","shares","buy","sell","price","rating","target",
        "today","news","dow","points", ticker.lower()
    }
    all_text = " ".join(df["headline"]).lower()
    words = [w for w in all_text.split() if w.isalpha()]
    meaningful = [w for w in words if w not in stop_words and w not in ignore]
    return Counter(meaningful).most_common(7)
