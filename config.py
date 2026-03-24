import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent
DATA_RAW       = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MODELS_DIR     = ROOT_DIR / "models"
REPORTS_DIR    = ROOT_DIR / "reports"

API_KEY  = os.getenv("OPENAQ_API_KEY")
BASE_URL_OPENAQ = "https://api.openaq.org/v3"
HEADERS  = {"X-API-Key": API_KEY}
RATE_LIMIT_SLEEP = 1.1 # giây giữa các request — 60 req/min → ~1 giây/req

BASE_URL_OPENMETEO = "https://archive-api.open-meteo.com/v1/archive"


HCMC_LAT = 10.8231
HCMC_LON = 106.6297
SENSOR_ID   = 11357424
SENSOR_NAME = "CMT8"
DATE_START  = "2024-11-19"
DATE_END    = "2026-03-23"
# Train / validation / test split
TRAIN_START = "2024-11-19"
TRAIN_END   = "2025-10-31"
VAL_START   = "2025-11-01"
VAL_END     = "2025-12-31"
TEST_START  = "2026-01-01"
TEST_END    = "2026-03-22"


HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "boundary_layer_height",
]

# LAG_HOURS       = [1, 2, 3, 6, 12, 24, 48]
# ROLLING_WINDOWS = [3, 6, 12, 24]
