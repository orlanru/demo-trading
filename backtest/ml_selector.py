"""Selector transversal con ML clásico.

Diseño según las prácticas estándar en predicción transversal de retornos
(Gu-Kelly-Xiu para el planteamiento; López de Prado para la validación):

1. **Target transversal desmediado**: no predecimos el retorno del activo, sino
   su retorno MENOS la media del universo ese mes. Es lo que decide la elección,
   y elimina el factor de mercado común (que no aporta nada a una decisión de
   "cuál compro" y domina toda la varianza si no se quita).
2. **Features estandarizadas por corte transversal**: cada mes se convierten a
   rango dentro del universo. Hace el modelo inmune a cambios de nivel/régimen
   y evita que el escalado de un activo domine.
3. **Walk-forward con embargo**: entrenamiento sólo con meses anteriores, con un
   hueco de `embargo` meses entre train y test, porque el target a h meses se
   solapa con las últimas filas de entrenamiento.
4. **Modelos con poca varianza**: ~700 observaciones para 8 features. Ridge es
   la elección sensata; el GBM se incluye para comprobar que no aporta.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from backtest import signals as sig
from backtest.crosssec import Panel


def build_features(panel: Panel, trend_lookback: int = 36) -> pd.DataFrame:
    """Panel largo (mes, activo) -> features + target."""
    close = panel.close

    feats = {
        "trend_z": sig.trend_gap(close, lookback=trend_lookback),   # +alto = más barato
        "mom_12_1": sig.momentum_12_1(close, lookback=12, skip=1),
        "mom_6_1": sig.momentum_12_1(close, lookback=6, skip=1),
        "rev_1m": sig.reversal_1m(close),
        "dd_36": sig.drawdown_from_high(close, lookback=trend_lookback),
        "vol_12": close.pct_change().rolling(12).std(),
        "dist_sma12": close / close.rolling(12).mean() - 1,
        "ret_36": close / close.shift(36) - 1,
    }

    long = []
    for name, df in feats.items():
        s = df.stack(future_stack=True).rename(name)
        long.append(s)
    X = pd.concat(long, axis=1)
    X.index.names = ["month", "asset"]

    fwd = close.shift(-1) / close - 1
    y = fwd.stack(future_stack=True).rename("fwd")
    y.index.names = ["month", "asset"]
    df = X.join(y).dropna()

    # target transversal: exceso sobre la media del universo ese mes
    df["y"] = df["fwd"] - df.groupby(level="month")["fwd"].transform("mean")
    return df


def _cross_sectional_ranks(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Cada feature -> rango dentro del mes, reescalado a [-1, 1]."""
    out = df.copy()
    for c in cols:
        r = df.groupby(level="month")[c].rank(pct=True)
        out[c] = 2 * r - 1
    return out


def walk_forward_scores(
    panel: Panel,
    model: str = "ridge",
    warmup: int = 48,
    retrain_every: int = 12,
    embargo: int = 1,
    trend_lookback: int = 36,
    alpha: float = 10.0,
) -> pd.DataFrame:
    """Devuelve la matriz de puntuaciones predichas (mes x activo), sin lookahead."""
    df = build_features(panel, trend_lookback)
    fcols = [c for c in df.columns if c not in ("fwd", "y")]
    df = _cross_sectional_ranks(df, fcols)

    months = df.index.get_level_values("month").unique().sort_values()
    if len(months) <= warmup:
        raise ValueError(f"Sólo {len(months)} meses con features; warmup={warmup}")

    preds = {}
    est = None
    for i in range(warmup, len(months)):
        if est is None or (i - warmup) % retrain_every == 0:
            # embargo: se descartan los últimos `embargo` meses de entrenamiento
            train_months = months[: max(0, i - embargo)]
            tr = df[df.index.get_level_values("month").isin(train_months)]
            if model == "ridge":
                est = Ridge(alpha=alpha)
            elif model == "gbm":
                est = HistGradientBoostingRegressor(
                    max_depth=3, max_iter=200, learning_rate=0.05,
                    min_samples_leaf=20, l2_regularization=1.0, random_state=0,
                )
            else:
                raise ValueError(f"modelo desconocido: {model}")
            est.fit(tr[fcols], tr["y"])

        ts = months[i]
        cur = df[df.index.get_level_values("month") == ts]
        p = est.predict(cur[fcols])
        preds[ts] = pd.Series(p, index=cur.index.get_level_values("asset"))

    return pd.DataFrame(preds).T.reindex(columns=panel.close.columns)


def in_sample_r2(panel: Panel, model: str = "ridge", trend_lookback: int = 36,
                 alpha: float = 10.0) -> float:
    """R² dentro de muestra. Sirve para ver cuánto sobreajusta el modelo:
    si el R² in-sample es alto y el edge out-of-sample es cero, es memorización."""
    from sklearn.metrics import r2_score

    df = build_features(panel, trend_lookback)
    fcols = [c for c in df.columns if c not in ("fwd", "y")]
    df = _cross_sectional_ranks(df, fcols)
    est = Ridge(alpha=alpha) if model == "ridge" else HistGradientBoostingRegressor(
        max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=20,
        l2_regularization=1.0, random_state=0,
    )
    est.fit(df[fcols], df["y"])
    return float(r2_score(df["y"], est.predict(df[fcols])))
