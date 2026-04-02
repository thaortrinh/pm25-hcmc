# PM2.5 Forecasting App - Ho Chi Minh City

This project is a Streamlit app for live air-quality monitoring and short-term PM2.5 forecasting in HCMC.

## What the app does

- Fetches live PM2.5 history from OpenAQ
- Fetches live weather and gas pollutants from Open-Meteo
- Computes Vietnam AQI categories and primary pollutant
- Runs CatBoost multi-output inference for 6 future horizons (t+1 to t+6)
- Displays a Dashboard page for current conditions and a Prediction page for forecast output

## Current architecture (src-focused)

- `App.py`
  - Streamlit entry page.
- `pages/1_Dashboard.py`
  - Current pollutant/weather cards, AQI scale, and PM2.5 last-24h chart.
- `pages/2_Prediction.py`
  - Prediction workflow UI with:
    - API autofill
    - manual input overrides
    - 6 fixed forecast cards (shown as `--` before inference)
    - chart and text summary after clicking `Chay du bao`
- `pages/3_Settings.py`
  - Theme and chart color controls.

- `src/api.py`
  - Aggregates current data for Dashboard from OpenAQ + Open-Meteo and computes AQI.
- `src/aqi.py`
  - AQI breakpoints, nowcast logic, and VN AQI conversion.
- `src/ui.py`
  - Shared Streamlit CSS/theme utilities.

- `src/services/openaq_client.py`
  - OpenAQ hourly PM2.5 history loader.
- `src/services/openmeteo_client.py`
  - Open-Meteo hourly weather history loader.

- `src/inference/feature_builder.py`
  - Feature engineering, scaling reconstruction, and multi-horizon feature/target preparation.
- `src/inference/artifact.py`
  - Model artifact discovery, loading, and rebuild fallback.
- `src/inference/predict.py`
  - End-to-end forecast orchestration for UI.
- `src/inference/train_artifact.py`
  - CLI entrypoint to rebuild deployable artifacts.

Note: `src/model.py` is a legacy mock module and is not used by the active Streamlit prediction flow.

## Model artifacts

Primary artifact location used by inference:

- `notebooks/model/multi_6h_weights/catboost_multi_horizon_deployable.cbm`
- `notebooks/model/multi_6h_weights/deployment_metadata.json`

Training reference script:

- `notebooks/model/6h_pm.py`

Discovery fallback also scans:

- `notebooks/model/`
- `model/`
- `models/`

If no valid pair of model + metadata is found, the app can rebuild artifacts from local raw/processed CSV files.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Set your OpenAQ key in `.env`:

```env
OPENAQ_API_KEY=your_openaq_api_key
```

Optional runtime overrides:

```env
OPENAQ_SENSOR_ID=11357424
OPENMETEO_LAT=10.8231
OPENMETEO_LON=106.6297
```

## Run locally

Optional prebuild:

```bash
python -m src.inference.train_artifact
```

Run Streamlit:

```bash
streamlit run App.py
```

## Data files

| File                                     | Purpose                                   |
| ---------------------------------------- | ----------------------------------------- |
| `data/raw/pm25_sensor_11357424.csv`      | Raw PM2.5 hourly observations             |
| `data/raw/weather_openmeteo.csv`         | Raw hourly weather history                |
| `data/processed/pm25_processed_data.csv` | Processed feature reference               |
| `notebooks/model/6h_pm.py`               | Multi-horizon CatBoost training reference |

## Inference assumptions and notes

- Feature timestamps stay UTC-aligned to match training-era processing.
- Horizon count is fixed at 6 in the current deployed pipeline.
- Upstream provider latency can cause apparent "data lag" on Dashboard (for example, PM2.5 source updates later than weather).
- Streamlit caching (`ttl=300`) is used for API-backed calls.
