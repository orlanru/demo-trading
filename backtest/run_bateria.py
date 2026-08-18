import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from backtest.panels import load_panel, window_table, window_table_weights
from backtest.selection_test import selection_edge, newey_west_t
from backtest import signals as sig, plan_mensual as pm

def paridad_riesgo(close, lookback=36, iters=300):
    """Pesos con contribución al riesgo IGUAL para todos (usa correlaciones)."""
    rets = close.pct_change()
    out = {}
    cols = list(close.columns); A = len(cols)
    for i, ts in enumerate(close.index):
        if i < lookback: continue
        h = rets.iloc[i-lookback+1:i+1].dropna(how="any")
        if len(h) < lookback//2: continue
        S = h.cov().to_numpy()
        w = np.full(A, 1.0/A)
        for _ in range(iters):                      # algoritmo multiplicativo estándar
            mrc = S @ w                             # riesgo marginal
            rc = w * mrc                            # contribución al riesgo
            if (mrc <= 0).any() or rc.sum() <= 0: break
            w = w * (rc.mean() / np.maximum(rc, 1e-18)) ** 0.1
            w = np.clip(w, 1e-6, None); w = w / w.sum()
        out[ts] = pd.Series(w, index=cols)
    return pd.DataFrame(out).T

def tendencia_con_caja(close, lookback=12):
    """Sólo invierte en los que están sobre su media móvil; si ninguno, caja.
    Se devuelve como pesos que pueden sumar <1: el resto se queda sin invertir."""
    sma = close.rolling(lookback).mean()
    dentro = (close > sma).astype(float)
    n = dentro.sum(axis=1).replace(0, np.nan)
    return dentro.div(n, axis=0).fillna(0.0)        # suma 1 o 0 (todo a caja)

def evaluar(tk, nombre, ventana=24):
    p = load_panel(tk); cols = p.close.columns
    ew = pd.DataFrame(0.0, index=p.months, columns=cols)
    reglas = {
        "equiponderado":        None,
        "volatilidad inversa":  pm.pesos_objetivo(p.close, 12, 0.02, 0.40),
        "paridad de riesgo":    paridad_riesgo(p.close),
        "tendencia + caja":     tendencia_con_caja(p.close),
    }
    filas=[]
    for lab, w in reglas.items():
        t = window_table(p, ew, window=ventana) if w is None else window_table_weights(p, w, window=ventana)
        x = t["equipond"] if w is None else t["estrategia"]
        filas.append({"cartera":lab,"ret. medio %":100*x.mean(),"desv %":100*x.std(),
                      "peor %":100*x.min(),"ret/riesgo":x.mean()/x.std()})
    print("="*76); print(f"### {nombre} | {len(tk)} activos | {len(p.months)} meses"); print("="*76)
    print(pd.DataFrame(filas).to_string(index=False,float_format=lambda v:f"{v:,.2f}"))

    # señales de selección sobre el mismo universo
    ed=[]
    for s in ["trend_gap","drawdown","momentum_12_1","value_5y","combo_mom_value"]:
        sc = sig.make_scores(s, p.close)
        e = selection_edge(p, sc, horizon=1)
        if len(e)<20: continue
        x=e.to_numpy()
        ed.append({"señal":s,"n":len(x),"edge %/mes":100*x.mean(),"t":newey_west_t(x,1)})
    if ed:
        print("\n  selección de UN activo:")
        print("  " + pd.DataFrame(ed).to_string(index=False,float_format=lambda v:f"{v:,.2f}").replace("\n","\n  "))
    print()

DIVERSO=["SPY","EFA","EEM","IEF","TLT","LQD","HYG","GLD","DBC","VNQ"]
DIVERSO2=DIVERSO+["TIP","EWJ","FXI"]
evaluar(DIVERSO,"DIVERSO (10 clases de activo)")
evaluar(DIVERSO2,"DIVERSO ampliado (13)")
evaluar(["SPY","QQQ","IWM","GLD","SLV","NVDA","MSFT"],"tu cesta (referencia)")
