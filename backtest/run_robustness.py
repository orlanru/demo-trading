"""Estudio completo de selección transversal sobre histórico largo ajustado.

Reproduce todos los resultados de docs/seleccion-transversal.md:

    python -m backtest.run_robustness                 # tabla de ventanas + edge
    python -m backtest.run_robustness --regimes       # edge por régimen de mercado
    python -m backtest.run_robustness --ml            # modelos ML
    python -m backtest.run_robustness --all
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from backtest import signals as sig
from backtest.crosssec import Panel
from backtest.panels import (
    UNIVERSES, load_panel, market_drawdown, random_band, window_table,
)
from backtest.selection_test import newey_west_t, selection_edge, summarize

warnings.filterwarnings("ignore")

SIGNALS = ["trend_gap", "trend_gap_filtered", "drawdown", "reversal_1m", "momentum_12_1"]


def _fmt(v):
    return f"{v:,.1f}"


def tabla_ventanas(panel: Panel, window: int, n_random: int) -> None:
    rows = []
    for s in SIGNALS:
        sc = sig.make_scores(s, panel.close)
        t = window_table(panel, sc, window=window)
        d = t["estrategia"] - t["equipond"]
        dm = t["estrategia"] - t["mediana"]
        rows.append({
            "señal": s, "ventanas": len(t),
            "ret. medio %": 100 * t["estrategia"].mean(),
            "vs equipond (pp)": 100 * d.mean(),
            "% gana equipond": 100 * (d > 0).mean(),
            "vs mediana (pp)": 100 * dm.mean(),
            "% gana mediana": 100 * (dm > 0).mean(),
            "percentil medio": 100 * t["percentil"].mean(),
        })

    rb = random_band(panel, window=window, n=n_random)
    rows.append({
        "señal": f"ALEATORIA ({n_random} sorteos)", "ventanas": np.nan,
        "ret. medio %": np.nan,
        "vs equipond (pp)": rb["vs_equipond_medio"],
        "% gana equipond": rb["gana_equipond"],
        "vs mediana (pp)": np.nan, "% gana mediana": np.nan,
        "percentil medio": rb["percentil_medio"],
    })

    ew = pd.DataFrame(0.0, index=panel.months, columns=panel.close.columns)
    t0 = window_table(panel, ew, window=window)
    d0 = t0["equipond"] - t0["mediana"]
    rows.append({
        "señal": "(equiponderado)", "ventanas": len(t0),
        "ret. medio %": 100 * t0["equipond"].mean(),
        "vs equipond (pp)": 0.0, "% gana equipond": np.nan,
        "vs mediana (pp)": 100 * d0.mean(), "% gana mediana": 100 * (d0 > 0).mean(),
        "percentil medio": np.nan,
    })

    print(pd.DataFrame(rows).to_string(index=False, float_format=_fmt, na_rep="—"))
    print(f"\n  Banda del azar (5-95%) vs equiponderado: "
          f"[{rb['p5']:+.1f}, {rb['p95']:+.1f}] pp")
    print("  Una señal dentro de esa banda no se distingue de elegir al azar.\n")


def tabla_edge(panel: Panel) -> None:
    rows = []
    for s in SIGNALS:
        sc = sig.make_scores(s, panel.close)
        for h in (1, 3):
            rows.append(summarize(selection_edge(panel, sc, horizon=h), s, h))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print()


def tabla_regimenes(panel: Panel) -> None:
    dd = market_drawdown("SPY")
    rows = []
    for s in SIGNALS:
        sc = sig.make_scores(s, panel.close)
        e = selection_edge(panel, sc, horizon=1)
        if not len(e):
            continue
        d = dd.reindex(e.index)
        x = e.to_numpy()
        for label, mask in [
            ("TODO", np.ones(len(x), bool)),
            ("mercado en máximos (dd > -5%)", (d > -0.05).to_numpy()),
            ("corrección (-5% a -10%)", ((d <= -0.05) & (d > -0.10)).to_numpy()),
            ("caída > 10%", (d <= -0.10).to_numpy()),
        ]:
            xs = x[np.nan_to_num(mask, nan=False).astype(bool)]
            if len(xs) < 8:
                continue
            rows.append({"señal": s, "régimen": label, "n": len(xs),
                         "edge medio %": 100 * xs.mean(),
                         "t (NW)": newey_west_t(xs, 1),
                         "% aciertos": 100 * np.mean(xs > 0)})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print()


def tabla_ml(panel: Panel, window: int, warmup: int) -> None:
    from backtest.ml_selector import build_features, walk_forward_scores

    n_obs = len(build_features(panel))
    print(f"  observaciones para el modelo: {n_obs:,}")
    rows = []
    for model in ("ridge", "gbm"):
        sc = walk_forward_scores(panel, model=model, warmup=warmup, embargo=1)
        for h in (1, 3):
            rows.append(summarize(selection_edge(panel, sc, horizon=h), f"ML-{model}", h))
        t = window_table(panel, sc, window=window)
        d = t["estrategia"] - t["equipond"]
        rows.append({"señal": f"ML-{model} (ventanas {window}m)", "horizonte": "—",
                     "n": len(t), "edge medio %": 100 * d.mean(),
                     "t (Newey-West)": np.nan, "% aciertos": 100 * (d > 0).mean()})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:,.2f}",
                                       na_rep="—"))
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universes", default=",".join(UNIVERSES))
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--random-draws", type=int, default=200)
    ap.add_argument("--ml-warmup", type=int, default=60)
    ap.add_argument("--regimes", action="store_true")
    ap.add_argument("--ml", action="store_true")
    ap.add_argument("--edge", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    for name in args.universes.split(","):
        tickers = UNIVERSES[name]
        panel = load_panel(tickers)
        print("=" * 108)
        print(f"### {name} | {len(tickers)} activos | {panel.months[0]:%Y-%m} → "
              f"{panel.months[-1]:%Y-%m} ({len(panel.months)} meses)")
        print("=" * 108)

        print(f"\n-- Ventanas de {args.window} aportaciones mensuales --")
        tabla_ventanas(panel, args.window, args.random_draws)

        if args.edge or args.all:
            print("-- Edge de selección mes a mes (Newey-West) --")
            tabla_edge(panel)
        if args.regimes or args.all:
            print("-- Edge por régimen de mercado --")
            tabla_regimenes(panel)
        if args.ml or args.all:
            print("-- Modelos ML (walk-forward con embargo) --")
            tabla_ml(panel, args.window, args.ml_warmup)


if __name__ == "__main__":
    main()
