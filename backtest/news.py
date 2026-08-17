"""Señal de sentimiento de noticias a partir de GDELT.

La teoría de la opinión contraria dice: compra el activo peor tratado por la
prensa, porque el pesimismo se exagera y luego revierte. Para poder
cuantificarlo hacen falta noticias HISTÓRICAS, no un scrapeo de hoy — de ahí
GDELT, que publica el tono medio diario de la cobertura que casa con una
consulta, desde 2017.

Aviso metodológico importante
-----------------------------
El tono absoluto NO es comparable entre activos: la cobertura de "Nvidia" y la
de "silver price" tienen líneas base de tono distintas por el vocabulario del
tema, no por optimismo del mercado. Comparar niveles crudos entre activos mide
sobre todo de qué se está hablando.

Por eso la señal correcta es el tono **normalizado contra la propia historia de
cada activo** (z-score con ventana móvil): mide "qué tan mal tratado está este
activo PARA LO QUE ES NORMAL EN ÉL", que es lo que la teoría contraria necesita.
Se implementan las dos y se comparan, para que se vea la diferencia.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

NEWS_DIR = Path(__file__).resolve().parent.parent / "data" / "news"

# Consulta por activo: el subyacente, no el ticker (el ticker apenas sale en prensa)
QUERIES = {
    "SPY": '("S&P 500" OR "SP 500 index")',
    "QQQ": '("Nasdaq 100" OR "Nasdaq composite")',
    "IWM": '("Russell 2000" OR "small cap stocks")',
    "GLD": '("gold price" OR "gold prices")',
    "SLV": '("silver price" OR "silver prices")',
    # Sin paréntesis en los términos sueltos: GDELT sólo los admite alrededor
    # de expresiones con OR y devuelve error de sintaxis en otro caso.
    "NVDA": 'Nvidia',
    "MSFT": 'Microsoft',
}


def load_tone(tickers: list[str]) -> pd.DataFrame:
    """Tono medio mensual por activo (media de los tonos diarios del mes)."""
    series = {}
    for t in tickers:
        path = NEWS_DIR / f"{t}_timelinetone.json"
        if not path.exists():
            continue
        data = json.load(open(path))["timeline"][0]["data"]
        s = pd.Series(
            [d["value"] for d in data],
            index=pd.to_datetime([d["date"] for d in data], format="%Y%m%dT%H%M%SZ"),
        )
        series[t] = s.resample("ME").mean()
    if not series:
        raise FileNotFoundError(
            f"No hay datos de noticias en {NEWS_DIR}. Ejecuta: python -m backtest.fetch_news"
        )
    return pd.DataFrame(series).dropna(how="all")


def load_volume(tickers: list[str]) -> pd.DataFrame:
    """Volumen de cobertura mensual (% de artículos que casan la consulta)."""
    series = {}
    for t in tickers:
        path = NEWS_DIR / f"{t}_timelinevol.json"
        if not path.exists():
            continue
        data = json.load(open(path))["timeline"][0]["data"]
        s = pd.Series(
            [d["value"] for d in data],
            index=pd.to_datetime([d["date"] for d in data], format="%Y%m%dT%H%M%SZ"),
        )
        series[t] = s.resample("ME").mean()
    return pd.DataFrame(series).dropna(how="all") if series else pd.DataFrame()


# --------------------------------------------------------------------------
# Señales. Convención del proyecto: puntuación más alta = más preferido.
# "Contraria" = preferir al peor tratado -> se invierte el signo del tono.
# --------------------------------------------------------------------------

def tone_z(tone: pd.DataFrame, lookback: int = 12) -> pd.DataFrame:
    """Tono normalizado contra la propia historia de cada activo.

    Durante el calentamiento (sin `lookback` meses de historia) devuelve NaN,
    que es correcto: no se puede normalizar sin referencia. Una serie sin
    varianza alguna devuelve 0 (no hay anomalía), no NaN.
    """
    mu = tone.rolling(lookback).mean()
    sd = tone.rolling(lookback).std()
    z = (tone - mu) / sd.where(sd > 0)
    return z.mask((sd == 0) & mu.notna(), 0.0)


def news_contrarian_z(tone: pd.DataFrame, lookback: int = 12) -> pd.DataFrame:
    """LA IDEA: comprar el activo con el tono más deprimido respecto a su normal."""
    return -tone_z(tone, lookback)


def news_contrarian_level(tone: pd.DataFrame, **_) -> pd.DataFrame:
    """Versión ingenua: comparar niveles de tono crudos entre activos.

    Se incluye para demostrar por qué está mal: mide el vocabulario del tema,
    no el pesimismo del mercado.
    """
    return -tone


def news_momentum_z(tone: pd.DataFrame, lookback: int = 12) -> pd.DataFrame:
    """Dirección opuesta: comprar al mejor tratado por la prensa."""
    return tone_z(tone, lookback)


def news_drop(tone: pd.DataFrame, **_) -> pd.DataFrame:
    """Comprar al que MÁS ha empeorado su tono este mes (choque de pesimismo)."""
    return -(tone - tone.shift(1))


def news_attention_drop(tone: pd.DataFrame, volume: pd.DataFrame,
                        lookback: int = 12) -> pd.DataFrame:
    """Pesimismo ponderado por atención: sólo cuenta si además se habla mucho.

    Un tono malo con cobertura testimonial es ruido; la teoría contraria supone
    una masa de inversores leyendo lo mismo.
    """
    z = tone_z(tone, lookback)
    sd = volume.rolling(lookback).std()
    # cobertura perfectamente plana -> no hay anomalía de atención, no NaN
    vz = ((volume - volume.rolling(lookback).mean()) / sd.where(sd > 0)).fillna(0.0)
    return (-z) * vz.clip(lower=0).reindex_like(z)


REGISTRY = {
    "news_contrarian_z": news_contrarian_z,
    "news_contrarian_level": news_contrarian_level,
    "news_momentum_z": news_momentum_z,
    "news_drop": news_drop,
}


def make_scores(name: str, tone: pd.DataFrame, volume: pd.DataFrame | None = None,
                lookback: int = 12) -> pd.DataFrame:
    if name == "news_attention_drop":
        if volume is None or volume.empty:
            raise ValueError("news_attention_drop necesita datos de volumen")
        return news_attention_drop(tone, volume, lookback)
    return REGISTRY[name](tone, lookback=lookback)
