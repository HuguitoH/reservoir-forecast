# Reservoir Forecast — Madrid Water System

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-6.x-purple?logo=plotly&logoColor=white)
![CI](https://github.com/HuguitoH/reservoir-forecast/actions/workflows/ci.yml/badge.svg)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B?logo=streamlit&logoColor=white)](https://reservoir-forecast.streamlit.app/)

Time series forecasting of the Madrid reservoir system capacity.
Built across three notebooks covering ETL, EDA and modelling, with a production-ready
Streamlit dashboard for operational drought risk monitoring.

**Dataset:** Canal de Isabel II · 13 reservoirs · monthly capacity (hm³) · 1998–2021

---

## App

Live at **[reservoir-forecast.streamlit.app](https://reservoir-forecast.streamlit.app/)**

Four pages designed around operational decision-making:

**System Overview** — current system status, configurable XGBoost forecast (3 months to 12 years),
drought risk assessment with month-by-month classification.

**Model Comparison** — all four models evaluated on the test period 2017–2021, feature importance
analysis, forecast vs actual chart.

**Drought Analysis** — month × year heatmap coloured by drought severity, historical drought
months per year, summary statistics.

**Reservoir Explorer** — individual reservoir capacity, average monthly profile,
contribution to the system total.

---

## Results

| Model       | R²    | MAE (hm³) | RMSE (hm³) | MAPE  | Accuracy |
|-------------|-------|-----------|------------|-------|----------|
| **XGBoost** | **0.82** | **31.1** | **45.4** | **4.6%** | **95.4%** |
| LSTM        | 0.73  | 44.4      | 55.1       | 6.7%  | 93.3%    |
| SARIMAX     | 0.30  | 74.7      | 89.1       | 11.9% | 88.1%    |
| Prophet     | -0.43 | 103.4     | 127.0      | 17.2% | 82.8%    |

XGBoost selected for production — best metrics, 726 KB serialised, fast inference,
interpretable feature importance. `rolling_mean_3` and `lag_1` account for 70% of
predictive power — the system has strong short-term momentum.

Train: Jan 1998 – Dec 2016 (228 months) · Test: Jan 2017 – Mar 2021 (51 months)

> [!Note]
> XGBoost outperforms Bidirectional LSTM with only 279 data points — well-engineered
> lag and rolling features beat deep learning on small datasets.

---

## Project Structure

```
reservoir-forecast/
├── app.py                          # Streamlit app — System Overview
├── pages/
│   ├── 01_Models.py                # Model comparison
│   ├── 02_Drought.py               # Drought analysis
│   └── 03_Reservoirs.py            # Reservoir explorer
├── notebooks/
│   ├── 01_etl.ipynb                # Load, clean, export processed data
│   ├── 02_eda.ipynb                # Seasonality, stationarity, ACF/PACF
│   └── 03_modelling.ipynb          # Prophet, SARIMAX, LSTM, XGBoost
├── src/
│   ├── etl.py                      # load_raw_csvs, clean_dataframe, build_pivot
│   ├── features.py                 # build_xgb_features
│   ├── evaluate.py                 # compute_metrics
│   └── forecast.py                 # iterative_forecast, classify_capacity
├── tests/
│   ├── test_etl.py
│   ├── test_features.py
│   ├── test_evaluate.py
│   └── test_forecast.py
├── icons/
├── models/
│   ├── xgb_bundle.pkl              # XGBoost + scalers + forecast + metadata
│   ├── lstm_bundle.pkl             # Bidirectional LSTM weights + scalers
│   ├── sarimax_bundle.pkl          # SARIMAX fitted model
│   └── prophet_bundle.pkl          # Prophet fitted model
├── data/
│   └── processed/
│       ├── reservoirs_pivot.csv    # Wide-format pivot (279 rows × 17 cols)
│       ├── reservoirs_long.csv     # Long-format clean data
│       └── eda_summary.json        # Drought thresholds + stationarity results
└── pyproject.toml
```

### Why this structure?

**`src/`** contains four modules with single responsibilities — ETL, feature engineering,
evaluation metrics, and iterative forecast. The notebooks import from `src/` and only
orchestrate — no business logic inline.

**`tests/`** — 33 unit tests with 98% coverage on `src/`. Pure unit tests with synthetic
fixtures and `tmp_path` for I/O — no dependency on real CSV data.

**Three notebooks, one direction** — notebook 01 exports processed CSVs, notebook 02
exports `eda_summary.json`, notebook 03 imports both and exports trained models.
Running out of order will fail intentionally.

---

## Notebook 01 — ETL

Raw data from Canal de Isabel II: 14 CSV files, one per reservoir, with BOM-prefixed
column names, European decimal commas, and year only appearing in January rows.

Key decisions:

**BOM handling** — `utf-8-sig` encoding strips the BOM automatically. Latin-1 fallback
for files with European characters.

**Year reconstruction** — `RioLozoya_Riosequillo` has all year values missing.
Reconstructed from the known 1998–2021 sequence. All other reservoirs use forward-fill
per reservoir group.

**`RioLosMorales_LosMorales` excluded** — data ends in 2018, 3 years short of the rest.

**Truncation to March 2021** — all reservoirs end at this date.

---

## Notebook 02 — EDA

**Seasonality** — peak capacity in April–May (snowmelt + spring rainfall), trough in
September–October (end of dry summer). Strong, stable annual cycle.

**Drought thresholds** derived from historical distribution:
- Severe: < 488 hm³ (p10) — 28 months in the record (10%)
- Moderate: < 574 hm³ (p25) — 42 additional months (15%)

**Stationarity** — ADF test: p = 0.000034. Series is stationary — no differencing needed
for SARIMAX.

**ACF/PACF** — PACF dominant at lag 1 (0.92), cuts off at lag 2. Implies AR(2) non-seasonal
component. Justifies SARIMAX(2,0,1)(1,1,1,12).

---

## Notebook 03 — Modelling

Four models trained on identical train/test splits. Selection criterion: MAPE + R².

**Prophet** — captures seasonal shape but overestimates amplitude. MAPE 17.2%.

**SARIMAX(2,0,1)(1,1,1,12)** — order derived from ACF/PACF. Correct seasonal structure
but confidence interval widens rapidly beyond 12 months. MAPE 11.9%.

**Bidirectional LSTM** — window size 24, lag features + cyclic month encoding.
77 epochs, val_loss 0.0027. MAPE 6.7%.

**XGBoost** — 12 lag features + rolling statistics (mean/std over 3, 6, 12 months) +
cyclic month encoding. Feature importance: `rolling_mean_3` (0.41) + `lag_1` (0.29)
= 70% of predictive power. MAPE 4.6%.

---

## Stack

- Python 3.12
- XGBoost, scikit-learn
- TensorFlow / Keras — Bidirectional LSTM
- statsmodels — SARIMAX
- Prophet — Facebook Prophet
- Plotly — interactive visualisations
- pandas, numpy, scipy
- Streamlit — 4-page operational dashboard
- pytest + pytest-cov — 33 unit tests, 98% coverage

---

## Run Locally

```bash
git clone https://github.com/HuguitoH/reservoir-forecast
cd reservoir-forecast
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run tests
pytest

# Run notebooks in order — each exports files needed by the next
# notebooks/01_etl.ipynb
# notebooks/02_eda.ipynb
# notebooks/03_modelling.ipynb

# Run the app
streamlit run app.py
```

> [!TIP]
> Run the notebooks in order before launching the app.
> Notebook 01 exports the processed CSVs, notebook 02 exports `eda_summary.json`,
> notebook 03 exports the model PKLs. The app will fail if any of these are missing.

Raw CSV data from Canal de Isabel II — place the 14 reservoir CSV files in `data/raw/`
before running notebook 01.

---

## Known Limitations

> [!WARNING]
> This model was trained on Canal de Isabel II data from 1998–2021.
> It is not generalisable to other reservoir systems or geographies.

- Iterative forecast accumulates error at each step — beyond 24 months treat as a plausible scenario, not a precise estimate
- `RioLozoya_ElAtazar` accounts for 47% of system capacity — the total forecast is essentially a forecast of one reservoir
- Dataset ceiling reached at 279 monthly observations — more data would improve all models
