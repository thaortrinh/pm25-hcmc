# PM2.5 Prediction — Ho Chi Minh City

Forecasting PM2.5 air pollution levels in Ho Chi Minh City using sensor data from OpenAQ combined with historical weather data from Open-Meteo.

---

Data was collected from `2024-11-19` to `2026-03-23` 

We build an end-to-end pipeline including:

- Data collection from APIs
- Data preprocessing & feature engineering
- Exploratory Data Analysis (EDA)
- Machine learning modeling
- Deploy model with Streamlit UI

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Collection](#data-collection)
- [Data](#data)
- [Roadmap](#roadmap)

---

## Overview

This project builds an end-to-end pipeline to collect, process, and predict PM2.5 concentrations in Ho Chi Minh City. The goal is to train a forecasting model using meteorological features (temperature, humidity, wind speed, etc.) alongside historical air quality readings.

**Data sources:**

This project using sensor data from OpenAQ combined with historical weather data from Open-Meteo.

- **PM2.5:** [OpenAQ API](https://openaq.org/) - sensor readings from HCMC monitoring stations
- **Weather:** [Open-Meteo API](https://open-meteo.com/) - hourly historical weather data (free, no API key required)

---

# Project Structure

```bash
pm25-prediction/
│
├── data/
│   ├── raw/
│   │   ├── hcmc_stations.csv
│   │   ├── pm25_sensor_11357424.csv
│   │   └── weather_openmeteo.csv
│   │
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── data/
│   │   ├── collect_openaq.py
│   │   ├── collect_openmeteo.py
│   │   ├── merge.py
│   │
│   └── __init__.py
│
├── config.py
├── .env
├── requirements.txt
└── README.md
```


---

## Installation

```bash
# Clone the repository
git clone https://github.com/thaortrinh/pm25-hcmc.git
cd pm25-prediction

# Create a virtual environment
python -m venv venv

source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

**Configure `.env`** (if using an OpenAQ API key):

```env
OPENAQ_API_KEY=your_api_key
```

---

## Data

I've already fetch raw data and put them in folder `data/`:
| File | Description | Source |
|------|-------------|--------|
| `hcmc_stations.csv` | Station metadata (ID, coordinates, name) | OpenAQ |
| `pm25_sensor_11357424.csv` | Hourly PM2.5 time series (μg/m³) | OpenAQ |
| `weather_openmeteo.csv` | Temperature, humidity, wind speed, precipitation, etc. | Open-Meteo |

---
Or, you can **run scripts in order:**

```bash
# 1. Fetch station list & PM2.5 readings
python src/data/collect_openaq.py

# 2. Fetch corresponding weather data
python src/data/collect_openmeteo.py
```

Outputs are saved to `data/raw/`

---