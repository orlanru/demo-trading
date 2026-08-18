"""Evalúa el vault OKF (repo poc_OKF) como recomendador de inversión.

El vault publica llamadas direccionales con corpus congelado point-in-time, así
que se puede puntuar contra precios posteriores sin lookahead. Dos cortes:
2025-12-31 (20 llamadas) y 2026-03-31 (100 casuísticas, 60 direccionales).

Cuidado con el control: la cartera que sale del OKF suele quedar muy
concentrada, porque descarta todo lo que marca 'desfavorable'. Compararla contra
carteras aleatorias DIVERSIFICADAS infla el percentil — una apuesta concentrada
tiene mucha más dispersión. El control correcto son carteras aleatorias del
mismo tamaño.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from backtest.panels import load_panel

# Llamadas del sistema A, corpus congelado a 2026-03-31
LLAMADAS_2026Q1 = {
    "QQQ": ("Nasdaq Composite", "desfavorable"), "SPY": ("S&P 500", "desfavorable"),
    "DIA": ("Dow Jones", "desfavorable"),        "EWJ": ("Nikkei 225", "desfavorable"),
    "EFA": ("Euro Stoxx 50", "desfavorable"),    "FXI": ("Hang Seng", "desfavorable"),
    "SLV": ("Plata", "desfavorable"),            "GLD": ("Oro", "desfavorable"),
    "XLF": ("US Banks", "desfavorable"),         "NVDA": ("Nvidia", "favorable"),
    "DBC": ("Petróleo WTI", "favorable"),
}
PESO = {"favorable": 1.0, "neutra": 0.5, "desfavorable": 0.0}


def evaluar(llamadas: dict, asof: str) -> dict:
    tk = list(llamadas)
    c = load_panel(tk).close
    base = c.index[c.index <= asof][-1]
    ret = (c.loc[c.index[-1]] / c.loc[base] - 1).reindex(tk).to_numpy()

    dirs = [llamadas[t][1] for t in tk]
    acierto = sum((d == "favorable" and r > 0) or (d == "desfavorable" and r < 0)
                  for d, r in zip(dirs, ret))

    w = np.array([PESO[d] for d in dirs])
    elegidos = int((w > 0).sum())
    r_okf = float((w @ ret) / w.sum()) if w.sum() > 0 else float("nan")

    # control correcto: todas las carteras del MISMO número de activos
    combos = np.array([np.mean(ret[list(ix)])
                       for ix in combinations(range(len(tk)), elegidos)])

    return {
        "asof": str(base.date()), "hasta": str(c.index[-1].date()),
        "n_llamadas": len(tk), "aciertos": acierto,
        "activos_elegidos": elegidos,
        "ret_okf": r_okf, "ret_equipond": float(ret.mean()),
        "percentil_vs_mismo_tamaño": 100 * float((combos < r_okf).mean()),
        "retornos": pd.Series(ret, index=tk),
    }


if __name__ == "__main__":
    r = evaluar(LLAMADAS_2026Q1, "2026-03-31")
    print(f"\nOKF as-of {r['asof']} → {r['hasta']}")
    print(f"  aciertos direccionales : {r['aciertos']}/{r['n_llamadas']} "
          f"({100*r['aciertos']/r['n_llamadas']:.0f}%)  |  moneda: 50%")
    print(f"  cartera resultante     : {r['activos_elegidos']} de {r['n_llamadas']} activos")
    print(f"  retorno OKF            : {100*r['ret_okf']:+.1f}%")
    print(f"  retorno equiponderado  : {100*r['ret_equipond']:+.1f}%")
    print(f"  percentil vs carteras aleatorias del mismo tamaño: "
          f"{r['percentil_vs_mismo_tamaño']:.0f}\n")
