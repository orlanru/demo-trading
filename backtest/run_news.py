"""¿Comprar el activo peor tratado por las noticias bate al DCA?

Teoría de la opinión contraria: el pesimismo de la prensa se exagera y luego
revierte, así que el peor tratado sería la mejor compra.

Uso:
    python -m backtest.run_news                # todo
    python -m backtest.run_news --diagnostico  # sólo el test de causalidad
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from backtest import news
from backtest.crosssec import Panel
from backtest.panels import load_panel, random_band, window_table
from backtest.selection_test import newey_west_t, selection_edge, summarize

warnings.filterwarnings("ignore")

SIGNALS = ["news_contrarian_z", "news_contrarian_level", "news_momentum_z", "news_drop"]


def diagnostico_causalidad(panel: Panel, tone: pd.DataFrame) -> None:
    """¿El tono de las noticias ADELANTA al precio, o sólo lo SIGUE?

    Si el tono correlaciona fuerte con el retorno PASADO y nada con el FUTURO,
    entonces "comprar el peor tratado" es un momentum inverso mal implementado,
    no información nueva. Es la crítica que decide si la idea puede funcionar.
    """
    close = panel.close.reindex(columns=tone.columns).dropna(how="all")
    idx = close.index.intersection(tone.index)
    close, t = close.loc[idx], tone.loc[idx]

    ret = close.pct_change()
    z = news.tone_z(t, 12)

    filas = []
    for lag, etiqueta in [(-2, "retorno de hace 2 meses"),
                          (-1, "retorno del mes anterior"),
                          (0, "retorno del MISMO mes"),
                          (1, "retorno del mes SIGUIENTE"),
                          (2, "retorno a 2 meses vista")]:
        # shift(-lag) alinea con el mes actual: lag<0 trae el pasado, lag>0 el futuro
        r = ret.shift(-lag)
        pares = pd.concat([z.stack().rename("z"), r.stack().rename("r")], axis=1).dropna()
        if len(pares) < 30:
            continue
        c = float(pares["z"].corr(pares["r"]))
        # t aproximado de la correlación
        n = len(pares)
        tstat = c * np.sqrt((n - 2) / max(1e-9, 1 - c ** 2))
        filas.append({"tono vs": etiqueta, "n": n, "correlación": c, "t": tstat})

    print("  Correlación entre el tono de las noticias y el retorno del activo:")
    print(pd.DataFrame(filas).to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print("\n  Si la correlación es alta con el pasado y ~0 con el futuro, el tono")
    print("  es un espejo del precio: no aporta información nueva.\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="SPY,QQQ,IWM,GLD,SLV,NVDA,MSFT")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--lookback", type=int, default=12)
    ap.add_argument("--random-draws", type=int, default=200)
    ap.add_argument("--diagnostico", action="store_true")
    args = ap.parse_args()

    tickers = args.tickers.split(",")
    tone = news.load_tone(tickers)
    volume = news.load_volume(tickers)
    disponibles = list(tone.columns)

    panel = load_panel(disponibles)
    idx = panel.months.intersection(tone.index)
    panel = Panel(buy=panel.buy.loc[idx], close=panel.close.loc[idx])
    tone = tone.loc[idx]
    if not volume.empty:
        volume = volume.reindex(idx)

    print(f"\nActivos con noticias: {', '.join(disponibles)}")
    print(f"Periodo: {idx[0]:%Y-%m} → {idx[-1]:%Y-%m} ({len(idx)} meses)\n")

    print("-- Diagnóstico: ¿el tono adelanta o sigue al precio? --")
    diagnostico_causalidad(panel, tone)
    if args.diagnostico:
        return

    señales = {}
    for s in SIGNALS:
        señales[s] = news.make_scores(s, tone, volume, lookback=args.lookback)
    if not volume.empty:
        try:
            señales["news_attention_drop"] = news.make_scores(
                "news_attention_drop", tone, volume, lookback=args.lookback)
        except Exception as e:
            print(f"  (news_attention_drop no disponible: {e})")

    print("-- Edge de selección mes a mes (Newey-West) --")
    filas = [summarize(selection_edge(panel, sc, horizon=h), n, h)
             for n, sc in señales.items() for h in (1, 3)]
    print(pd.DataFrame(filas).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print()

    print(f"-- Ventanas de {args.window} aportaciones vs DCA equiponderado --")
    filas = []
    for n, sc in señales.items():
        t = window_table(panel, sc, window=args.window)
        d = t["estrategia"] - t["equipond"]
        dm = t["estrategia"] - t["mediana"]
        filas.append({"señal": n, "ventanas": len(t),
                      "ret. medio %": 100 * t["estrategia"].mean(),
                      "vs equipond (pp)": 100 * d.mean(),
                      "% gana equipond": 100 * (d > 0).mean(),
                      "vs mediana (pp)": 100 * dm.mean(),
                      "% gana mediana": 100 * (dm > 0).mean(),
                      "percentil medio": 100 * t["percentil"].mean()})

    rb = random_band(panel, window=args.window, n=args.random_draws)
    filas.append({"señal": f"ALEATORIA ({args.random_draws})", "ventanas": np.nan,
                  "ret. medio %": np.nan,
                  "vs equipond (pp)": rb["vs_equipond_medio"],
                  "% gana equipond": rb["gana_equipond"],
                  "vs mediana (pp)": np.nan, "% gana mediana": np.nan,
                  "percentil medio": rb["percentil_medio"]})
    print(pd.DataFrame(filas).to_string(index=False, float_format=lambda v: f"{v:,.1f}",
                                        na_rep="—"))
    print(f"\n  Banda del azar (5-95%): [{rb['p5']:+.1f}, {rb['p95']:+.1f}] pp\n")


if __name__ == "__main__":
    main()
