# ¿La orden límite del predictor mejora al DCA?

**Respuesta corta: no, y no es culpa del modelo.** El mecanismo pierde contra DCA
en SPY y en BTC-USD, con cualquier variante de ejecución que probemos, y el techo
teórico de la idea es demasiado pequeño para cubrir el error de predicción.

Este documento explica cómo se midió y por qué.

---

## 1. Qué se ha medido

El repo no tenía baseline: la app calcula un nivel de compra pero nunca se
comparó con la alternativa obvia (comprar y ya está). El módulo `backtest/`
cubre ese hueco.

Reglas de la simulación (todas las estrategias aportan **el mismo dinero**,
$1.000 al mes; lo único que cambia es a qué precio entra y si llega a entrar):

| Estrategia | Regla |
|---|---|
| `DCA` | compra en la primera apertura de cada mes, siempre |
| `LIMIT (acumula caja)` | orden límite en `pred_low + margen`; si no ejecuta, el dinero espera al mes siguiente |
| `LIMIT (fallback cierre)` | igual, pero si no ejecuta se compra al cierre del mes |
| `LIMIT (fallback open t+1)` | igual, pero si no ejecuta se compra en la apertura del mes siguiente |
| `ORACLE (mínimo real)` | previsión perfecta: el límite es **exactamente** el mínimo real del mes |

`ORACLE` es la clave del análisis: es la **cota superior de toda la
arquitectura**. Ningún modelo, por bueno que sea, puede superarla, porque nadie
compra más barato que el mínimo del mes.

Como los `.pkl` no están versionados, `backtest/model.py` reentrena la misma
receta de la app (`0.7·XGBoost + 0.3·Ridge` sobre las features mensuales de
`services/features.py`, objetivo `low_t / close_{t-1} - 1`) pero **walk-forward**:
en el mes *t* sólo se usa información disponible al cierre de *t-1*, con
reentreno anual. Sin esto cualquier resultado sería lookahead.

Reproducir:

```bash
python -m backtest.run --ticker SPY     --warmup 36 --sweep-naive --offline
python -m backtest.run --ticker BTC-USD --warmup 36 --sweep-naive --offline
python -m backtest.test_engine   # 7 tests del simulador
```

---

## 2. Resultados

### SPY — 70 meses (2020-11 → 2026-08), $1.000/mes

Límite ejecutado en el **41,4 %** de los meses. MAE de la predicción: **3,26 pp**.

| Estrategia | % ejec. | Coste medio | Valor final | TIR anual | vs DCA |
|---|---|---|---|---|---|
| **DCA** | 100 % | 481,69 | **112.818** | 16,69 % | — |
| LIMIT (acumula caja) | 41,4 % | 495,40 | 108.562 | 15,33 % | **−3,77 %** |
| LIMIT (fallback cierre) | 41,4 % | 485,69 | 111.891 | 16,40 % | **−0,82 %** |
| LIMIT (fallback open t+1) | 41,4 % | 483,69 | 111.748 | 16,35 % | **−0,95 %** |
| ORACLE (mínimo real) | 100 % | 465,14 | 116.834 | 17,92 % | *+3,56 %* |

### BTC-USD — 83 meses (2019-10 → 2026-08), $1.000/mes

Límite ejecutado en el **31,3 %** de los meses. MAE: **9,18 pp**.

| Estrategia | % ejec. | Coste medio | Valor final | TIR anual | vs DCA |
|---|---|---|---|---|---|
| **DCA** | 100 % | 25.807 | **206.957** | 26,58 % | — |
| LIMIT (acumula caja) | 31,3 % | 28.232 | 186.622 | 23,55 % | **−9,83 %** |
| LIMIT (fallback cierre) | 31,3 % | 26.792 | 199.350 | 25,48 % | **−3,68 %** |
| LIMIT (fallback open t+1) | 31,3 % | 26.602 | 199.353 | 25,48 % | **−3,67 %** |
| ORACLE (mínimo real) | 100 % | 22.144 | 241.197 | 31,08 % | *+16,54 %* |

---

## 3. Por qué no mejora

### 3.1. El premio no compone; el coste de fallar, sí

El descuento medio capturable dentro de un mes (de la apertura al mínimo) es:

| | descuento medio open→low | retorno mensual medio |
|---|---|---|
| SPY | 3,25 % | +1,17 % |
| BTC-USD | 12,76 % | +6,12 % |

Comprar un 3,25 % más barato **esa aportación concreta** te da un 3,25 % más de
participaciones de **ese mes**, y ahí se acaba: el descuento se aplica una vez,
no compone. Por eso el oráculo de SPY sólo saca +3,56 % en casi 6 años (≈0,6 %
anual) pese a acertar el mínimo exacto 70 veces seguidas.

Fallar un mes, en cambio, sí compone: ese dinero se pierde toda la deriva del
activo durante el resto del horizonte. Ese es exactamente el −3,77 % (SPY) y
−9,83 % (BTC) de la variante que acumula caja.

**El techo es lineal y pequeño; el riesgo es compuesto y grande.**

### 3.2. El error de predicción se come el premio entero

| | premio total disponible | MAE del modelo |
|---|---|---|
| SPY | 3,25 % | 3,26 pp |
| BTC-USD | 12,76 % | 9,18 pp |

En SPY el error de predicción es **igual** al premio total. No hay margen: para
capturar algo habría que acertar el mínimo mensual con un error mucho menor que
3 pp, y eso no lo hace nadie. En BTC el premio es mayor (la volatilidad ayuda),
pero el error también, y se lleva el 72 %.

### 3.3. No es el modelo: el mecanismo pierde solo

Sustituyendo el modelo por una regla ingenua (`límite = cierre anterior × (1−k)`),
el resultado es **monótono**: cuanto más esperas, peor.

**SPY**

| k | % ejec. | vs DCA |
|---|---|---|
| 0 % | 90,0 % | −0,33 % |
| 1 % | 68,6 % | −0,65 % |
| 2 % | 50,0 % | −1,02 % |
| 3 % | 38,6 % | −1,81 % |
| 5 % | 22,9 % | −1,79 % |
| 7 % | 15,7 % | −3,86 % |
| 10 % | 4,3 % | −5,94 % |

**BTC-USD**

| k | % ejec. | vs DCA |
|---|---|---|
| 0 % | 100 % | 0,00 % |
| 1 % | 92,8 % | +0,15 % |
| 2 % | 85,5 % | +0,06 % |
| 3 % | 83,1 % | −0,02 % |
| 5 % | 69,9 % | −3,87 % |
| 7 % | 56,6 % | −5,20 % |
| 10 % | 42,2 % | −3,86 % |

Ni siquiera k=0 (esperar a que el precio vuelva al cierre del mes anterior) bate
al DCA. Los +0,15 % de BTC con k=1 % son ruido, no señal. Cambiar de modelo no
arregla esto porque el modelo no es el problema.

### 3.4. El camino de mejora converge... al DCA

El objetivo de error cuadrático de la app estima la **media condicional** del
mínimo, así que la orden se queda corta la mitad de las veces (de ahí el 41 %
de ejecución). Se arregla con pérdida pinball, apuntando a un cuantil alto del
mínimo (`--quantile`):

| cuantil | SPY % ejec. | SPY vs DCA | BTC % ejec. | BTC vs DCA |
|---|---|---|---|---|
| 0,50 | 48,6 % | −1,00 % | 28,9 % | −4,92 % |
| 0,80 | 80,0 % | −0,63 % | 56,6 % | −2,95 % |
| 0,95 | 92,9 % | **−0,10 %** | 79,5 % | **−0,95 %** |

*(variante fallback-cierre)*

Funciona: mejora de forma monótona. Pero fíjate a dónde lleva: cuanto más subes
el cuantil, más se ejecuta la orden, y el límite converge hacia "compra ya".
En el límite q→1 la estrategia **es** el DCA, y llega a él **desde abajo**.
Optimizar este sistema es optimizar el camino de vuelta al baseline.

### 3.5. Problemas de parametrización (secundarios, pero reales)

- **`margin_usd` es un importe fijo en dólares.** Para SPY son $5: eran el 2,3 %
  del precio en 2016 y son el 0,64 % hoy. El margen de seguridad se estrecha
  solo según se revaloriza el activo, sin que nadie lo decida. Debería ser un
  porcentaje (`--margin-pct` en el backtest lo prueba: sube la ejecución del
  41 % al 60 %, aunque sigue perdiendo, −0,46 %).
- **BTC tiene `margin_usd = 0`**: cero tolerancia sobre el activo cuyo error de
  predicción es de 9 pp.
- **Muestra insuficiente para XGBoost.** El objetivo es mensual: ~120 filas para
  SPY desde 2016, ~310 si arrancas en 2000, con 14 features y árboles. La
  relación observaciones/parámetros garantiza sobreajuste; el backtest
  walk-forward es lo que lo revela.

---

## 4. Qué haría falta para batir al DCA

Nada de lo anterior se arregla afinando el `.pkl`. Las salidas reales son:

1. **Aceptar el DCA como baseline y competir en otra dimensión** (fiscalidad,
   rebalanceo entre activos, aportaciones extra en drawdowns profundos medidos
   en desviaciones típicas, no en predicciones de mínimos).
2. **Cambiar el horizonte.** El premio intra-mes no compone. Una señal que
   decida *cuánto* aportar (no *a qué precio*) sí compone, porque cambia la
   exposición sostenida.
3. **Si se mantiene la orden límite**, usar cuantil alto (q≈0.9) con fallback
   obligatorio a cierre de mes: es la configuración que menos pierde
   (−0,1 % en SPY). Es decir, un DCA con un pequeño descuento oportunista, no
   una estrategia de timing.
4. **Reposicionar el producto.** La app es un buen visualizador de niveles y
   contexto macro/sentimiento. Como generador de alfa sobre DCA, los datos
   dicen que no.

---

## 5. Limitaciones del estudio

Honestidad sobre lo que este backtest no prueba:

- **Ventana alcista.** 2020-2026 (SPY) y 2019-2026 (BTC) son periodos de
  tendencia alcista fuerte, el peor escenario posible para esperar caídas. En un
  mercado lateral o bajista prolongado la orden límite lo haría mejor en
  términos relativos. El argumento del §3.1 (el premio no compone) se mantiene
  igualmente, pero la magnitud de la derrota sería menor.
- **Precios sin ajustar por dividendos** (fuente: Nasdaq para SPY, Coinbase para
  BTC, cacheados en `data/cache/` porque Yahoo devuelve 429 desde el entorno de
  CI). Esto **favorece a la estrategia límite**, no al DCA: el DCA mantiene más
  participaciones durante más tiempo, así que contando dividendos su ventaja
  sería mayor. La conclusión es conservadora.
- **Sin comisiones ni slippage**, y se asume ejecución al precio límite exacto
  cuando el mínimo lo toca. También favorece a la estrategia límite (el DCA
  paga las mismas comisiones, pero la variante límite se beneficia de una
  ejecución idealizada).
- **El modelo no es tu `.pkl`.** Es la misma arquitectura reentrenada
  walk-forward. Si tu `.pkl` está entrenado sobre todo el histórico, sus
  resultados en pantalla serán mejores que estos, pero por lookahead. En
  cualquier caso, el `ORACLE` acota a **cualquier** modelo, incluido el tuyo.
- **Warmup de 36 meses** para dejar ventana de test utilizable con 10 años de
  datos. Con `--warmup 60` la conclusión no cambia (SPY: −1,17 % la variante
  acumula-caja, oráculo +3,02 %).
