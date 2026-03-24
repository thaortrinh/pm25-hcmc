import requests
import pandas as pd
import numpy as np
from pathlib import Path

from config import (
    BASE_URL_OPENMETEO, HOURLY_VARIABLES,
    DATA_RAW,
    HCMC_LAT, HCMC_LON,
    DATE_START, DATE_END,
)

# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_weather(lat: float, lon: float, date_start: str, date_end: str) -> dict:
    """
    Gọi Open-Meteo Historical API, trả về raw JSON response.
    """
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "start_date":      date_start,
        "end_date":        date_end,
        "hourly":          ",".join(HOURLY_VARIABLES),
        "wind_speed_unit": "ms",      # m/s thay vì km/h mặc định
        "timezone":        "GMT",     # giữ UTC để đồng bộ với OpenAQ data
    }

    print(f"Fetching Open-Meteo: {date_start} → {date_end}...")
    resp = requests.get(BASE_URL_OPENMETEO, params=params, timeout=60)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Open-Meteo trả về lỗi {resp.status_code}:\n{resp.text}"
        )

    data = resp.json()

    if data.get("error"):
        raise RuntimeError(f"Open-Meteo error: {data.get('reason')}")

    return data


# ── Parse ─────────────────────────────────────────────────────────────────────

def parse_response(data: dict) -> pd.DataFrame:
    """
    Chuyển raw JSON thành DataFrame.

    Response có dạng:
      data["hourly"]["time"]           → list timestamps
      data["hourly"]["temperature_2m"] → list values
      ...

    Tất cả arrays cùng length, index tương ứng nhau.
    """
    hourly = data["hourly"]

    df = pd.DataFrame({"datetime": hourly["time"]})

    for var in HOURLY_VARIABLES:
        if var in hourly:
            df[var] = hourly[var]
        else:
            # Variable không có trong response — điền NaN, không crash
            df[var] = np.nan
            print(f"  [!] Variable '{var}' không có trong response")

    df["datetime"] = pd.to_datetime(df["datetime"])   # format: "2024-11-19T00:00"
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


# ── Feature bổ sung ───────────────────────────────────────────────────────────

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính thêm features từ variables gốc.

    wind_u, wind_v: thành phần gió — quan trọng hơn speed+direction vì
    giữ được thông tin hướng dưới dạng continuous variable cho model.
    """
    wind_rad = np.radians(df["wind_direction_10m"])
    df["wind_u"] = -df["wind_speed_10m"] * np.sin(wind_rad)   # thành phần Đông-Tây
    df["wind_v"] = -df["wind_speed_10m"] * np.cos(wind_rad)   # thành phần Bắc-Nam

    return df


# ── Quality report ────────────────────────────────────────────────────────────

def quality_report(df: pd.DataFrame) -> None:
    print(f"\n{'='*50}")
    print(f"Quality report: Open-Meteo weather")
    print(f"{'='*50}")
    print(f"Khoảng thời gian: {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"Tổng giờ        : {len(df):,}")

    print(f"\nMissing theo variable:")
    for col in HOURLY_VARIABLES:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            pct = n_missing / len(df) * 100
            status = " ← cần kiểm tra" if pct > 0 else ""
            print(f"  {col:<30}: {n_missing:>5} ({pct:.1f}%){status}")

    print(f"\nSample values (3 hàng đầu):")
    print(df[["datetime"] + HOURLY_VARIABLES].head(3).to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    out_path = DATA_RAW / "weather_openmeteo.csv"

    if out_path.exists():
        print(f"[skip] File đã tồn tại: {out_path.name}")
        print("Xóa file nếu muốn download lại.")
        return

    raw_data = fetch_weather(HCMC_LAT, HCMC_LON, DATE_START, DATE_END)
    df       = parse_response(raw_data)
    df       = add_derived_features(df)

    quality_report(df)

    df.to_csv(out_path, index=False)
    print(f"\nĐã lưu → {out_path}")
    print(f"Shape  : {df.shape}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()