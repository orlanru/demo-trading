"""Modelo walk-forward que replica la arquitectura de la app.

La app carga un `.pkl` ya entrenado (`0.7*XGBoost + 0.3*Ridge` sobre features
mensuales) y no se versiona en el repo, así que aquí lo reentrenamos con la
misma receta pero **walk-forward**: en cada mes t sólo se usa información
disponible hasta el cierre de t-1. Eso es lo que permite medir el edge real.

Objetivo (igual que la app): pred_drop = low_t / close_{t-1} - 1
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from services.features import build_monthly_features

FEATURES = [
    "RV_1m", "RangeMean_1m", "RangeStd_1m", "DownDaysFrac_1m", "MaxDD_1m",
    "Retorno_1m", "Retorno_3m", "Retorno_6m", "Retorno_12m",
    "Dist_SMA12", "RSI_14", "Range_Pct", "Mes_sin", "Mes_cos",
]


def build_dataset(dfd: pd.DataFrame) -> pd.DataFrame:
    """Une features de t-1 con el target del mes t."""
    m = build_monthly_features(dfd).dropna()
    ds = m[FEATURES].copy()
    ds["prev_close"] = m["Close"]
    # target: caída del mes SIGUIENTE respecto al cierre de este mes
    ds["target_low"] = m["Low"].shift(-1)
    ds["y"] = ds["target_low"] / ds["prev_close"] - 1
    ds["target_month"] = m.index.to_series().shift(-1)
    return ds.dropna()


def _fit(X: pd.DataFrame, y: pd.Series, quantile: float | None):
    if quantile is None:
        xgb = XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=0, n_jobs=1,
        )
    else:
        xgb = XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=quantile,
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=0, n_jobs=1,
        )
    xgb.fit(X, y)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, y)
    return xgb, ridge


def walk_forward_predictions(
    dfd: pd.DataFrame,
    warmup: int = 60,
    retrain_every: int = 12,
    quantile: float | None = None,
    debias: bool | None = None,
) -> pd.DataFrame:
    """Predice `pred_drop` mes a mes sin mirar el futuro.

    warmup        -> meses mínimos de entrenamiento antes de la primera predicción
    retrain_every -> cada cuántos meses se reentrena (12 = anual)
    quantile      -> None usa error cuadrático (como la app); 0.8 usa pérdida
                     pinball, que apunta al percentil 80 del mínimo mensual.
                     En modo cuantil se usa sólo el XGB (el Ridge es L2 y
                     arrastraría la predicción de vuelta a la media).
    debias        -> resta el sesgo medio del entrenamiento (el `bias` del .pkl).
                     Por defecto activo en modo media, desactivado en modo
                     cuantil (recentrar a la media anula el cuantil).
    """
    if debias is None:
        debias = quantile is None
    w_xgb = 1.0 if quantile is not None else 0.7
    ds = build_dataset(dfd)
    if len(ds) <= warmup:
        raise ValueError(f"Histórico insuficiente: {len(ds)} meses, warmup={warmup}")

    rows = []
    models = None
    bias = 0.0

    for i in range(warmup, len(ds)):
        if models is None or (i - warmup) % retrain_every == 0:
            train = ds.iloc[:i]
            models = _fit(train[FEATURES], train["y"], quantile)
            if debias:
                pred_tr = (w_xgb * models[0].predict(train[FEATURES])
                           + (1 - w_xgb) * models[1].predict(train[FEATURES]))
                bias = float(np.mean(train["y"] - pred_tr))
            else:
                bias = 0.0

        row = ds.iloc[[i]]
        p = (w_xgb * models[0].predict(row[FEATURES])[0]
             + (1 - w_xgb) * models[1].predict(row[FEATURES])[0])
        pred_drop = float(p + bias)

        rows.append(
            {
                "target_month": row["target_month"].iloc[0],
                "prev_close": float(row["prev_close"].iloc[0]),
                "pred_drop": pred_drop,
                "pred_low": float(row["prev_close"].iloc[0]) * (1 + pred_drop),
                "actual_low": float(row["target_low"].iloc[0]),
                "actual_drop": float(row["y"].iloc[0]),
            }
        )

    out = pd.DataFrame(rows).set_index("target_month")
    out.index = out.index.to_period("M").to_timestamp("M")  # alinear con fin de mes
    return out
