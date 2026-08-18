"""Regla de aportación mensual para una cesta concreta.

No predice qué activo subirá — eso ya se probó y no funciona. Hace lo único que
sí sobrevivió a las pruebas: equilibrar el RIESGO en vez de los euros, y dirigir
la aportación de cada mes a lo que se haya quedado por detrás de su objetivo.

El resultado es que cada mes sí te dice dónde meter el dinero, y normalmente
concentra en uno o dos activos. Pero el motivo no es una predicción: es que ésos
son los que están por debajo de su peso objetivo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pesos_objetivo(close: pd.DataFrame, lookback: int = 12,
                   suelo: float = 0.05, techo: float = 0.30) -> pd.DataFrame:
    """Pesos objetivo inversamente proporcionales a la volatilidad.

    Con topes: sin ellos, el activo más tranquilo se come la cartera y el más
    volátil desaparece, que no es lo que quieres si lo tienes por algo.
    """
    vol = close.pct_change().rolling(lookback).std()
    inv = 1.0 / vol.replace(0, np.nan)
    w = inv.div(inv.sum(axis=1), axis=0)

    n = w.shape[1]
    if n * suelo > 1 + 1e-9 or n * techo < 1 - 1e-9:
        raise ValueError(
            f"Topes imposibles con {n} activos: suelo={suelo}, techo={techo}. "
            f"Hace falta n*suelo <= 1 <= n*techo.")

    # Recortar y renormalizar de golpe vuelve a sacar los pesos del tope. Se
    # busca el factor lambda tal que sum(clip(w*lambda, suelo, techo)) = 1;
    # esa suma crece con lambda, así que la bisección siempre converge.
    def ajustar(fila):
        if not np.isfinite(fila).all():
            return fila
        lo, hi = 0.0, 1.0
        # se amplía el techo hasta que la suma recortada llegue a 1: con pesos
        # muy asimétricos el lambda necesario puede ser de varios órdenes
        for _ in range(200):
            if np.clip(fila * hi, suelo, techo).sum() >= 1.0:
                break
            hi *= 2
        for _ in range(80):
            mid = (lo + hi) / 2
            if np.clip(fila * mid, suelo, techo).sum() < 1.0:
                lo = mid
            else:
                hi = mid
        return np.clip(fila * (lo + hi) / 2, suelo, techo)

    W = np.array([ajustar(f) for f in w.to_numpy(dtype=float)])
    return pd.DataFrame(W, index=w.index, columns=w.columns)


def reparto_del_mes(valor_actual: pd.Series, objetivo: pd.Series,
                    aportacion: float) -> pd.Series:
    """Cómo repartir la aportación de este mes. Nunca vende.

    Mira cuánto le falta a cada activo para llegar a su peso objetivo sobre la
    cartera resultante, y reparte proporcionalmente a esa carencia. Si nadie
    está por debajo (raro), reparte según los pesos objetivo.
    """
    total = float(valor_actual.sum()) + aportacion
    falta = (objetivo * total - valor_actual).clip(lower=0)
    if falta.sum() <= 0:
        base = objetivo
    else:
        base = falta
    return aportacion * base / base.sum()


def simular(panel, lookback: int = 12, aportacion: float = 1000.0,
            suelo: float = 0.05, techo: float = 0.30,
            meses: pd.DatetimeIndex | None = None) -> dict:
    """Simula la regla completa sin lookahead: los pesos objetivo del mes t se
    calculan con datos hasta el cierre de t-1."""
    idx = panel.months if meses is None else meses
    obj = pesos_objetivo(panel.close, lookback, suelo, techo).reindex(idx)
    buy = panel.buy.loc[idx]
    close = panel.close.loc[idx]
    cols = list(panel.close.columns)

    shares = pd.Series(0.0, index=cols)
    historia = []
    for i, ts in enumerate(idx):
        precios = buy.loc[ts]
        valor = shares * precios
        o = obj.iloc[i - 1] if i > 0 else pd.Series(1 / len(cols), index=cols)
        if o.isna().any():
            o = pd.Series(1 / len(cols), index=cols)
        euros = reparto_del_mes(valor, o, aportacion)
        shares = shares + euros / precios
        historia.append({"mes": ts, "valor": float((shares * close.loc[ts]).sum()),
                         **{f"eur_{c}": euros[c] for c in cols}})
    return {"shares": shares,
            "valor_final": float((shares * close.iloc[-1]).sum()),
            "aportado": aportacion * len(idx),
            "objetivo_final": obj.iloc[-1],
            "historia": pd.DataFrame(historia).set_index("mes")}
