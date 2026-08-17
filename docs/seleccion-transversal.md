# DCA con selección de activo: ¿comprar el más barato bate al DCA?

Estudio de la idea: aportar una cantidad fija cada mes, pero en lugar de repartirla
siempre igual, meterla en **el activo que cotiza más barato respecto a su línea de
tendencia**.

**Resultado: no funciona. La estrategia acaba exactamente donde acaba elegir al
azar, y la señal tiene edge negativo.**

Datos: precios mensuales **ajustados por dividendos**, 4 universos, hasta 27 años
de histórico. Aportación de $1.000/mes, ventanas de 24 aportaciones.

```bash
python -m backtest.fetch_data                      # descarga los datos
python -m backtest.run_robustness --all            # reproduce todo lo de abajo
python -m backtest.test_panels                     # valida el motor vectorizado
```

| universo | activos | periodo | meses |
|---|---|---|---|
| `sectores` | 9 sectoriales SPDR + SPY | 1998-12 → 2026-08 | 333 |
| `cesta_usuario` | SPY, QQQ, IWM, GLD, SLV, NVDA, MSFT | 2006-04 → 2026-08 | 245 |
| `cesta_sin_nvda` | igual, sin NVDA | 2006-04 → 2026-08 | 245 |
| `amplio` | sectores + SPY, QQQ, IWM, GLD, TLT, EFA | 2004-11 → 2026-08 | 262 |

---

## 1. La respuesta: se coloca en la mediana, igual que el azar

Percentil medio de la estrategia entre los DCA individuales:

| universo | trend_gap (la idea) | elección aleatoria |
|---|---|---|
| sectores | 51,0 | 50,4 |
| cesta_usuario | 55,9 | 59,0 |
| cesta_sin_nvda | 50,5 | 51,4 |
| amplio | 48,5 | 49,0 |

Elegir el más barato coloca la inversión en el **percentil 50** de los activos
disponibles. Que es donde la coloca tirar un dado.

---

## 2. El benchmark propuesto no valía

La idea original era compararse con **la mediana de los DCA individuales**. Esa
vara es demasiado baja: la mediana no es una estrategia ejecutable — para
cobrarla tendrías que saber de antemano cuál va a quedar en el medio.

En la cesta original, `trend_gap` gana a la mediana el 62,2 % de las veces. Suena
bien hasta que ves que el **equiponderado**, sin decidir nada, la gana el 71,6 %.

**El benchmark correcto es el DCA equiponderado sobre el mismo universo**: mismo
dinero, mismos activos, cero decisiones.

---

## 3. Resultado contra el equiponderado

Diferencia media en puntos porcentuales sobre 222-310 ventanas de 24 aportaciones:

| señal | sectores | cesta_usuario | cesta_sin_nvda | amplio |
|---|---|---|---|---|
| trend_gap (la idea) | +0,5 | −1,6 | **−2,9** | −1,6 |
| trend_gap_filtered | +0,1 | +0,8 | −1,8 | −1,8 |
| drawdown (el más caído) | −1,2 | +6,9 | **−8,0** | **−5,8** |
| reversal_1m | −0,6 | +1,5 | −1,7 | −1,1 |
| momentum_12_1 | +0,4 | **+24,5** | +3,9 | +0,1 |
| **banda del azar (5-95 %)** | ±1,1 | ±4,2 | ±1,7 | ±1,2 |

Tres lecturas:

1. **`trend_gap` cae dentro o por debajo de la banda del azar** en los cuatro
   universos. En `cesta_sin_nvda` y `amplio` está por debajo: peor que el azar.
2. **Comprar el más caído (`drawdown`) es lo peor de todo**: −8,0 y −5,8 pp, muy
   fuera de la banda. No es ruido, es edge negativo.
3. **El +24,5 de momentum en la cesta original es NVDA.** La señal elige NVDA en
   70 de los 84 meses con señal. Quítala y queda +3,9. En el universo de 15
   activos queda +0,1.

NVDA está en la cesta porque en 2026 sabemos que multiplicó por 143. El sesgo no
está en la señal: está en **haber elegido el universo mirando hacia atrás**. Todo
estudio de este tipo debería hacer esta prueba — quitar el mejor activo y ver qué
queda.

---

## 4. El test con potencia estadística real

Las ventanas de 24 meses se solapan mes a mes: 300 ventanas no son 300
observaciones independientes, son unas 12. El test correcto mira la selección mes
a mes:

> Retorno del activo elegido − retorno medio del universo.
> Si la señal no informa, esa serie tiene media cero.

Errores estándar Newey-West. Con ~10 señales probadas, el umbral honesto de
significancia está en |t| ≈ 2,9, no en 2,0.

| señal | cesta_sin_nvda (245m) | sectores (333m) |
|---|---|---|
| trend_gap | −0,34 % (t = −1,17) | −0,12 % (t = −0,61) |
| trend_gap en mercado alcista | **−0,53 % (t = −1,70)** | −0,14 % (t = −0,70) |
| trend_gap_filtered | −0,47 % (t = −1,69) | −0,15 % (t = −0,79) |
| drawdown | −0,64 % (t = −1,70) | −0,33 % (t = −1,13) |
| drawdown en mercado alcista | **−0,88 % (t = −2,10)** | −0,20 % (t = −0,71) |
| momentum_12_1 | +0,27 % (t = 0,72) | +0,08 % (t = 0,33) |

Nada alcanza significancia — pero lo relevante es **el signo**: todas las
variantes de "comprar barato" son negativas, en los dos universos, en los dos
horizontes. La señal informa; informa al revés. La única con signo positivo es el
momentum, la dirección contraria, y es demasiado débil para afirmarla.

---

## 5. El ML tampoco

Le dimos su mejor escenario: universo ancho, muestra larga, target transversal
desmediado, features convertidas a rango dentro de cada mes, walk-forward con
embargo de 1 mes y reentreno anual.

| universo | obs. | modelo | R² dentro de muestra | edge fuera de muestra |
|---|---|---|---|---|
| sectores | 2.960 | Ridge | +0,004 | −0,12 % (t = −0,51) |
| sectores | 2.960 | GBM | **+0,108** | −0,22 % (t = −0,94) |
| amplio | 3.375 | Ridge | +0,004 | −0,41 % (t = −1,34) |
| amplio | 3.375 | GBM | **+0,108** | **−0,68 % (t = −2,46)** |
| cesta_sin_nvda | 1.248 | Ridge | +0,009 | −0,36 % (t = −1,28) |
| cesta_sin_nvda | 1.248 | GBM | **+0,174** | −0,02 % (t = −0,05) |

El GBM explica hasta el 17 % de la varianza dentro de muestra y transporta cero
—o menos que cero— fuera. Es memorización de ruido, el mismo problema del `.pkl`
del predictor original en otro escenario.

Y hay un techo estructural: 6-15 activos dan 6-15 opciones por mes. Los trabajos
de ML transversal que sí funcionan operan sobre **miles** de activos por corte.

---

## 6. La hipótesis del régimen: apareció y no validó

Mirando los datos surgió un patrón: `trend_gap` daba +1,12 %/mes (t = 1,85, 66 %
de aciertos) **sólo cuando el mercado ya estaba caído más del 10 %**.

Como esa hipótesis salió de mirar estos mismos datos, hay que validarla fuera de
donde apareció. Partiendo la serie por la mitad —explorar en la primera, juzgar
en la segunda:

- En sectores el signo **se invierte** entre mitades: −0,36 % → +1,30 %.
- La estrategia condicional completa (barato en caídas, equipeso el resto) pierde
  **50 pp** en sectores sobre 27 años.
- En `cesta_sin_nvda` el resultado cambia de signo según el umbral elegido:
  **+7,3 / −13,5 / −7,5 pp** para −5 % / −10 % / −15 %.

Un parámetro que voltea el signo del resultado es ruido, no un régimen.

---

## 7. Lo que sí aguanta

El **DCA equiponderado**: +478,6 % en 27 años en sectores, +503,4 % en 20 años en
la cesta sin NVDA. Ninguna de las 5 señales ni los 2 modelos lo bate de forma
consistente en los 4 universos.

Si se quiere seguir por aquí:

1. La única dirección con respaldo (débil) es la **contraria** a la intuición
   original: momentum, comprar el fuerte. Y aun así da +3,9 pp en un universo de
   6 activos y +0,1 en el de 15.
2. Donde sí hay margen real es en el **reparto**, no en la selección: volatilidad
   inversa o paridad de riesgo bajan la varianza sin predecir nada. No suben el
   retorno esperado, pero eso sí es replicable.

---

## 8. Limitaciones

- **Cesta elegida a posteriori.** NVDA y MSFT están ahí porque hoy sabemos que
  ganaron. El efecto está cuantificado en §3, no eliminado. Los universos
  `sectores` y `amplio` son menos vulnerables porque son sectoriales completos.
- **Sin comisiones, spreads ni fiscalidad.** La estrategia de selección concentra
  en 1 activo al mes y rota; el equiponderado no. En condiciones reales la
  estrategia quedaría **aún peor**.
- **Un periodo, aunque largo.** 1999-2026 incluye puntocom, 2008, covid y 2022,
  pero sigue siendo una única realización histórica.
- **Ventanas solapadas** en §3: sirven para ilustrar magnitudes, no para
  concluir. Las conclusiones se apoyan en §4, que sí tiene potencia.

---

## Fuentes

- [Cross-Sectional Equity Mean Reversion — QuantPedia](https://quantpedia.com/quantopian-quantpedia-trading-strategy-series-cross-sectional-equity-mean-rever/)
- [Cross-sectional momentum — pfolio academy](https://www.pfolio.io/academy/cross-sectional-momentum)
- [Dissecting Investment Strategies in the Cross Section and Time Series — CME Group](https://www.cmegroup.com/education/files/dissecting-investment-strategies-in-the-cross-section-and-time-series.pdf)
- [Purged cross-validation — Wikipedia](https://en.wikipedia.org/wiki/Purged_cross-validation)
- [Backtest Overfitting in the Machine Learning Era — Arian, Norouzi, Seco](https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID4686376_code4361537.pdf?abstractid=4686376&mirid=1)
- [The Deflated Sharpe Ratio — Bailey & López de Prado](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
- [8.3 The Dangers of Backtesting — Portfolio Optimization Book](https://portfoliooptimizationbook.com/book/8.3-dangers-backtesting.html)
