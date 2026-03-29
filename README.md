# PM2.5 Forecasting App - Ho Chi Minh City

This repository includes a deployable Streamlit inference pipeline for the CatBoost multi-horizon PM2.5 forecaster.

## What the deployment does

- Fetches recent PM2.5 sensor history from OpenAQ
- Fetches recent weather history from Open-Meteo
- Rebuilds the training-era engineered features used by `model/6h_pm.py`
- Loads a CatBoost multi-output artifact from `model/multi_6h_weights/`
- Generates a deployable CatBoost artifact if the repository only contains metrics and feature-importance files
- Shows a Vietnamese prediction UI with historical vs forecast PM2.5 charting

## Architecture

- `pages/2_Prediction.py`
  - Vietnamese Streamlit UI for autofill, manual overrides, horizon selection, results, and charting
- `src/services/openaq_client.py`
  - Recent hourly PM2.5 history loader from OpenAQ
- `src/services/openmeteo_client.py`
  - Recent hourly weather history loader from Open-Meteo
- `src/inference/feature_builder.py`
  - Rebuilds the base processed features, applies saved scaling, then adds the extra 6-hour CatBoost features from `model/6h_pm.py`
- `src/inference/artifact.py`
  - Detects the best serialized CatBoost artifact and creates one if missing
- `src/inference/predict.py`
  - End-to-end forecast orchestration for the UI
- `src/inference/train_artifact.py`
  - CLI entry point to materialize the deployable CatBoost artifact ahead of time

## Model artifact note

The repository currently contains:

- `model/multi_6h_weights/metrics.json`
- `model/multi_6h_weights/predictions.csv`
- `model/multi_6h_weights/feature_importance.csv`

It does not contain a serialized CatBoost model file. Because of that, the deployment code will generate:

- `model/multi_6h_weights/catboost_multi_horizon_deployable.cbm`
- `model/multi_6h_weights/deployment_metadata.json`

The deployment artifact keeps the CatBoost direct multi-horizon setup from `model/6h_pm.py` and excludes `target_next_hour`, which is a future-leaking helper column and should not be used at inference time.

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Then add your OpenAQ key to `.env`:

```env
OPENAQ_API_KEY=your_openaq_api_key
```

## Run locally

Optional prebuild step:

```bash
python -m src.inference.train_artifact
```

Start the app:

```bash
streamlit run App.py
```

## Data files already in the repository

| File | Purpose |
|------|---------|
| `data/raw/pm25_sensor_11357424.csv` | Raw PM2.5 hourly observations |
| `data/raw/weather_openmeteo.csv` | Raw hourly weather history |
| `data/processed/pm25_processed_data.csv` | Processed training-era feature reference |
| `model/6h_pm.py` | Multi-horizon CatBoost training script |

## Inference assumptions

- Time alignment stays in UTC for feature generation because the raw training pipeline also used UTC-aligned timestamps.
- `time_of_the_day`, rush-hour windows, dew-point approximation, and base scaling were reconstructed from repository artifacts and validated against `data/processed/pm25_processed_data.csv`.
- `special_holidays` uses a small inferred historical override set plus Vietnamese holidays when the `holidays` package is available.
