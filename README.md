# Quantitative Asset Terminal (Streamlit + Docker)

Aplicación en **Streamlit** para análisis cuantitativo con 3 módulos principales:
1) **Algorithmic Predictor**: calcula niveles de compra (AI projected low + limit buy) usando modelos `.pkl`.
2) **Macro Liquidity Monitor**: consulta indicadores macro desde **FRED** (St. Louis Fed).
3) **Sentiment & NLP Engine**: extrae titulares desde **Finviz**, calcula sentimiento con **VADER** y extrae keywords.

Incluye visualizaciones interactivas con **Plotly**: zoom/pan, hover detallado y range slider.

---

## Cómo funciona (resumen del flujo)

### 1) Algorithmic Predictor (SPY / BTC-USD)
- Carga un “brain” desde un `.pkl` (por ejemplo `modelo_final.pkl` o `modelo_btc.pkl`).
- Descarga precios con **yfinance**.
- Construye features mensuales (resampling, RV, drawdown, SMA, RSI, retornos, etc.).
- Usa el **mes cerrado anterior** para generar el input del modelo.
- Predice la caída/variación esperada (ensemble de XGBoost + Ridge).
- Calcula:
  - `base_low` (AI projected low)
  - `limit_order = base_low + margin_usd`
- Muestra métricas y un gráfico interactivo de precio/rango + niveles.

### 2) Macro Liquidity Monitor (FRED)
- Descarga series macro (ej. M2, balance de la Fed, HY spread, yield curve) desde FRED.
- Las muestra en gráficos Plotly con zoom y range slider.
- Cachea resultados (para no pedir FRED continuamente).

### 3) Sentiment & NLP Engine (Finviz + NLTK)
- Scrapea titulares del ticker en Finviz.
- Calcula sentiment de cada titular (compound score VADER).
- Saca “keywords” (stopwords + palabras ignoradas).
- Descarga precio intradía o diario del activo para comparar con el flujo de noticias.
- Muestra gráfico interactivo precio vs sentimiento + barra de keywords y tabla con titulares.

---

## ¿La estrategia mejora al DCA? (backtest)

**No.** El módulo `backtest/` compara la orden límite del predictor contra
comprar sin más (DCA), y pierde en SPY (−0,8 % a −3,8 %) y en BTC (−3,7 % a
−9,8 %) en todas las variantes probadas.

Lo importante no es el número, sino el techo: con **previsión perfecta** del
mínimo mensual la ganancia máxima sería de +3,6 % en SPY en 6 años, mientras que
el error del modelo es del mismo tamaño que ese premio. Y una regla ingenua sin
modelo pierde de forma monótona: cuanto más esperas, peor.

```bash
python -m backtest.run --ticker SPY     --warmup 36 --sweep-naive --offline
python -m backtest.run --ticker BTC-USD --warmup 36 --sweep-naive --offline
python -m backtest.test_engine
```

Análisis completo, metodología y limitaciones: **[docs/backtest-dca.md](docs/backtest-dca.md)**.

`--offline` usa el OHLC cacheado en `data/cache/`; sin esa opción se descarga de
yfinance como la app.

### Selección transversal: ¿y si elegimos qué activo comprar cada mes?

Segundo estudio: aportar lo mismo cada mes pero metiéndolo en el activo más
barato respecto a su línea de tendencia. Probado sobre **4 universos** (hasta 27
años, precios ajustados por dividendos).

**Tampoco funciona.** La estrategia acaba en el percentil 50 de los activos
disponibles — el mismo sitio que elegir al azar — y la señal tiene edge negativo
(−0,34 %/mes). Todo el edge aparente del universo original venía de tener NVDA en
la cesta. Incluye modelos ML (Ridge y GBM) con walk-forward y embargo, que
tampoco aportan nada.

```bash
python -m backtest.fetch_data                    # descarga datos ajustados
python -m backtest.run_robustness --all          # estudio completo
python -m backtest.test_panels
```

Análisis completo: **[docs/seleccion-transversal.md](docs/seleccion-transversal.md)**.

---

## Estructura del proyecto

Recomendada:

.
├─ app.py
├─ config.py
├─ requirements.txt
├─ Dockerfile
├─ models/
│  ├─ modelo_final.pkl
│  └─ modelo_btc.pkl
├─ services/
│  ├─ __init__.py
│  ├─ models.py
│  ├─ features.py
│  ├─ market_data.py
│  ├─ macro_data.py
│  └─ news_sentiment.py
├─ charts/
│  ├─ __init__.py
│  └─ plotly_charts.py
├─ backtest/
│  ├─ data.py           # OHLC (yfinance + cache CSV)
│  ├─ engine.py         # simulador DCA vs órdenes límite (1 activo)
│  ├─ model.py          # reentreno walk-forward de la arquitectura de la app
│  ├─ run.py            # CLI del estudio 1 (timing intra-mes)
│  ├─ crosssec.py       # simulador DCA con selección de activo
│  ├─ signals.py        # señales transversales (tendencia, momentum, reversión)
│  ├─ ml_selector.py    # selector ML walk-forward con embargo
│  ├─ selection_test.py # test de edge con errores Newey-West
│  ├─ panels.py         # paneles mensuales + motor vectorizado de ventanas
│  ├─ fetch_data.py     # descarga histórico largo ajustado por dividendos
│  ├─ run_crosssec.py   # CLI del estudio 2 (cache corta)
│  ├─ run_robustness.py # CLI del estudio 2 sobre histórico largo
│  ├─ test_engine.py
│  ├─ test_crosssec.py
│  └─ test_panels.py
├─ data/cache/          # OHLC diario cacheado (estudio 1)
├─ data/monthly/        # OHLC mensual ajustado, 20 activos (estudio 2)
└─ docs/
   ├─ backtest-dca.md
   └─ seleccion-transversal.md

> Importante: los modelos `.pkl` deben estar en `models/` (o ajusta rutas en `config.py`).

> **Los archivos `models/*.pkl` no están incluidos en este repositorio** (excluidos vía `.gitignore` por ser binarios pesados de ~6 MB cada uno). Ver la sección [Modelos `.pkl`](#modelos-pkl) para generarlos o conseguirlos.

---

## Requisitos

- Docker (recomendado), o
- Python 3.10+ para correr local

---

## Docker: pares de comandos para correrlo (build + run)

En Docker normalmente se usan **dos comandos**:

### A) BUILD (construir la imagen)
Crea una imagen a partir del Dockerfile (instala dependencias y copia tu proyecto dentro).

docker build -t quant-terminal .

- `-t quant-terminal` pone nombre (tag) a la imagen.
- El punto `.` indica que el contexto de build es la carpeta actual.

### B) RUN (ejecutar el contenedor)
Arranca un contenedor usando esa imagen y expone Streamlit al navegador.

docker run --rm -p 8501:8501 quant-terminal

- `-p 8501:8501` mapea el puerto del contenedor al puerto local.
- `--rm` borra el contenedor al cerrar (limpio para pruebas).

Luego abre:
http://localhost:8501

---

## Docker: modo desarrollo (hot reload)

Para desarrollar, puedes montar tu carpeta local dentro del contenedor. Así no necesitas rebuild en cada cambio:

### macOS / Linux
docker run --rm -p 8501:8501 -v "$(pwd)":/app quant-terminal

### Windows PowerShell
docker run --rm -p 8501:8501 -v ${PWD}:/app quant-terminal

---

## Ejecutar local (sin Docker)

1) Crear entorno virtual
python -m venv .venv

2) Activar

macOS/Linux:
source .venv/bin/activate

Windows PowerShell:
.\.venv\Scripts\Activate.ps1

3) Instalar dependencias
pip install -r requirements.txt

4) Ejecutar Streamlit
streamlit run app.py

Abrir:
http://localhost:8501

---

## Notas importantes

### Modelos `.pkl`
- SPY: models/modelo_final.pkl
- BTC-USD: models/modelo_btc.pkl

Si cambias nombres o ubicación: ajusta `config.py`.

Estos archivos **no se incluyen en el repositorio** (ver `.gitignore`) porque son binarios de varios MB, poco adecuados para versionar en git. Para correr la app necesitas:
1. Entrenar tus propios modelos (XGBoost + Ridge sobre los features generados por `services/features.py`) y guardarlos como `joblib.dump({"xgb_model": ..., "ridge_model": ..., "bias": ..., "best_iter": ..., "features": [...]}, "models/modelo_final.pkl")`, o
2. Colocar tus propios `.pkl` con esa misma estructura de diccionario en `models/`.

Sin estos archivos, la pestaña **ALGORITHMIC PREDICTOR** mostrará un error indicando que no encuentra el modelo; las demás pestañas (Macro Liquidity Monitor y Sentiment & NLP Engine) funcionan de forma independiente.

### NLTK en Docker
La app descarga recursos de NLTK (VADER + stopwords) si no existen.
Si tienes problemas de permisos o caché, puedes fijar la ruta NLTK:

En Dockerfile (opcional):
ENV NLTK_DATA=/app/nltk_data
RUN mkdir -p /app/nltk_data

---

## Troubleshooting

### 1) No carga la web
- Confirma puertos:
docker run --rm -p 8501:8501 quant-terminal
- Ver contenedores:
docker ps
- Ver logs:
docker logs <container_id>

### 2) ImportError con services/charts
Asegura que existen:
- services/__init__.py
- charts/__init__.py

### 3) No encuentra el modelo
Confirma que el archivo existe dentro del repo y se copia en la imagen:
- models/modelo_final.pkl
- models/modelo_btc.pkl

Si el modelo no se copia, revisa que no esté excluido por .dockerignore.

---

## Datos y limitaciones conocidas

- **yfinance**: datos de mercado con posible delay; sujeto a cambios/rate limits de Yahoo Finance.
- **FRED**: series macro públicas del St. Louis Fed, sin necesidad de API key para los endpoints usados aquí.
- **Finviz**: scraping de titulares vía HTML; puede romperse si Finviz cambia su estructura de página.
- Este proyecto es educativo/demostrativo. Los niveles calculados por el modelo **no son recomendaciones de inversión**.

---

## Licencia
MIT. Ver [LICENSE](LICENSE).
