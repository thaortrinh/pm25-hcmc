import os
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

load_dotenv()

from config import (
    API_KEY, BASE_URL_OPENAQ, HEADERS, RATE_LIMIT_SLEEP,
    DATA_RAW,
    HCMC_LAT, HCMC_LON,
    SENSOR_ID, SENSOR_NAME,
    DATE_START, DATE_END,
)

# ── Step 1: Fetch locations ──────────────────────

def fetch_locations() -> pd.DataFrame:
    """
    Trả về danh sách tất cả trạm PM2.5 trong bán kính 25km quanh TP.HCM.
    Dùng để verify và chọn sensor_id, không dùng trong pipeline chính.
    """
    url = f"{BASE_URL_OPENAQ}/locations"
    params = {
        "coordinates":   f"{HCMC_LAT},{HCMC_LON}",
        "radius":        25000,        # max theo API docs
        "parameters_id": 2,            # PM2.5
        "iso":           "VN",
        "limit":         100,
        "page":          1,
    }
    resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    results = resp.json().get("results", [])

    rows = []
    for loc in results:
        # Tìm sensor PM2.5 trong danh sách sensors của location
        pm25_sensors = [
            s for s in loc.get("sensors", [])
            if s["parameter"]["name"] == "pm25"
        ]
        for sensor in pm25_sensors:
            rows.append({
                "location_id":   loc["id"],
                "location_name": loc["name"],
                "sensor_id":     sensor["id"],
                "is_monitor":    loc["isMonitor"],
                "provider":      loc["provider"]["name"],
                "datetime_first": loc.get("datetimeFirst", {}).get("utc") if loc.get("datetimeFirst") else None,
                "datetime_last":  loc.get("datetimeLast",  {}).get("utc") if loc.get("datetimeLast")  else None,
                "distance_m":    round(loc.get("distance", 0)),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["datetime_first"] = pd.to_datetime(df["datetime_first"])
    df["datetime_last"]  = pd.to_datetime(df["datetime_last"])
    df = df.sort_values("datetime_last", ascending=False).reset_index(drop=True)
    return df


# ── Step 2: Fetch hourly measurements cho một sensor ─────────────────────────

def fetch_hours_one_month(
    sensor_id: int,
    date_from: datetime,
    date_to: datetime,
) -> list[dict]:
    """
    Lấy hourly data của một sensor trong khoảng [date_from, date_to).
    Tự động loop pagination nếu có nhiều hơn 1000 records/tháng.
    """
    url = f"{BASE_URL_OPENAQ}/sensors/{sensor_id}/hours"
    all_results = []
    page = 1

    while True:
        params = {
            "datetime_from": date_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "datetime_to":   date_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit":         1000,
            "page":          page,
        }

        resp = requests.get(url, params=params, headers=HEADERS, timeout=30)

        # Rate limit bị chặn
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("x-ratelimit-reset", 60))
            print(f"    Rate limit — chờ {retry_after}s...")
            time.sleep(retry_after)
            continue

        resp.raise_for_status()

        results = resp.json().get("results", [])
        all_results.extend(results)

        # Hết trang khi trả về ít hơn limit
        if len(results) < 1000:
            break

        page += 1
        time.sleep(RATE_LIMIT_SLEEP)

    return all_results


def parse_hours_response(results: list[dict]) -> pd.DataFrame:
    """
    Parse list results từ /hours endpoint thành DataFrame.

    Lấy các field:
    - period.datetimeFrom.utc  → timestamp của giờ đó
    - summary.avg              → PM2.5 trung bình trong giờ
    - summary.min/max/sd       → để EDA và quality check
    - coverage.percentComplete → % sensor hoạt động trong giờ đó
    """
    if not results:
        return pd.DataFrame()

    rows = []
    for r in results:
        period   = r.get("period", {})
        summary  = r.get("summary", {})
        coverage = r.get("coverage", {})

        dt_utc = period.get("datetimeFrom", {}).get("utc")
        if dt_utc is None:
            continue

        rows.append({
            "datetime":         dt_utc,
            "pm25_avg":         summary.get("avg"),
            "pm25_min":         summary.get("min"),
            "pm25_max":         summary.get("max"),
            "pm25_sd":          summary.get("sd"),
            "pm25_median":      summary.get("median"),
            "coverage_pct":     coverage.get("percentComplete"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    return df


def fetch_all_hours(
    sensor_id: int,
    sensor_name: str,
    date_start: str,
    date_end: str,
) -> pd.DataFrame:
    """
    Fetch toàn bộ hourly data từ date_start đến date_end, theo từng tháng.
    """
    start = datetime.strptime(date_start, "%Y-%m-%d")
    end   = datetime.strptime(date_end,   "%Y-%m-%d")

    all_dfs = []
    current = start

    while current < end:
        next_month = current + relativedelta(months=1)
        next_month = min(next_month, end)

        print(f"  [{sensor_name}] {current.strftime('%Y-%m')}...", end=" ", flush=True)

        results = fetch_hours_one_month(sensor_id, current, next_month)
        df_month = parse_hours_response(results)

        if df_month.empty:
            print("0 records")
        else:
            print(f"{len(df_month)} records")
            all_dfs.append(df_month)

        current = next_month
        time.sleep(RATE_LIMIT_SLEEP)

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    return df


# ── Step 3: Quality report ────────────────────────────────────────────────────

def quality_report(df: pd.DataFrame, sensor_name: str) -> None:
    """
    Kiểm tra missing rate, outlier, và coverage thấp.
    """
    if df.empty:
        print(f"[{sensor_name}] Không có data.")
        return

    # Tạo hourly index đầy đủ để đo missing rate thực sự
    full_idx = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="h")
    df_full  = df.set_index("datetime").reindex(full_idx)

    total   = len(df_full)
    missing = df_full["pm25_avg"].isna().sum()

    print(f"\n{'='*50}")
    print(f"Quality report: {sensor_name}")
    print(f"{'='*50}")
    print(f"Khoảng thời gian : {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"Tổng giờ          : {total:,}")
    print(f"Missing           : {missing:,} ({missing/total*100:.1f}%)")

    valid = df["pm25_avg"].dropna()
    print(f"PM2.5 avg  — min: {valid.min():.1f} | mean: {valid.mean():.1f} | max: {valid.max():.1f} µg/m³")

    # Giá trị bất thường
    n_negative = (valid < 0).sum()
    n_extreme  = (valid > 500).sum()
    n_low_cov  = (df["coverage_pct"] < 50).sum() if "coverage_pct" in df.columns else 0

    if n_negative > 0:
        print(f"[!] Giá trị âm    : {n_negative} giờ → set NaN trong bước cleaning")
    if n_extreme > 0:
        print(f"[!] Giá trị > 500 : {n_extreme} giờ → kiểm tra sensor error")
    if n_low_cov > 0:
        print(f"[!] Coverage < 50%: {n_low_cov} giờ → đánh dấu low quality")

    # Missing theo năm
    print(f"\nMissing theo năm:")
    yearly = df_full["pm25_avg"].isna().groupby(df_full.index.year).agg(["sum", "count"])
    yearly["pct"] = yearly["sum"] / yearly["count"] * 100
    for yr, row in yearly.iterrows():
        flag = "  ← nhiều missing" if row["pct"] > 40 else ""
        print(f"  {yr}: {row['pct']:.1f}%{flag}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not API_KEY:
        raise EnvironmentError(
            "Không tìm thấy OPENAQ_API_KEY.\n"
            "Tạo file .env ở root project với nội dung:\n"
            "OPENAQ_API_KEY=your_key_here"
        )

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    print("=== Danh sách trạm PM2.5 tại TP.HCM ===")
    df_locations = fetch_locations()
    if not df_locations.empty:
        print(df_locations[[
            "location_id", "sensor_id", "location_name",
            "is_monitor", "datetime_last", "distance_m"
        ]].to_string(index=False))
        df_locations.to_csv(DATA_RAW / "hcmc_stations.csv", index=False)
        print(f"\nĐã lưu → data/raw/hcmc_stations.csv\n")

    time.sleep(RATE_LIMIT_SLEEP)



    out_path = DATA_RAW / f"pm25_sensor_{SENSOR_ID}.csv"

    if out_path.exists():
        print(f"[skip] File đã tồn tại: {out_path.name}")
        return
    
    print(f"=== Download: {SENSOR_NAME} (sensor_id={SENSOR_ID}) ===")
    print(f"Range: {DATE_START} → {DATE_END}\n")

    df = fetch_all_hours(
        sensor_id   = SENSOR_ID,
        sensor_name = SENSOR_NAME,
        date_start  = DATE_START,
        date_end    = DATE_END,
    )

    if df.empty:
        print(f"[!] Không có data.")
        return

    quality_report(df, SENSOR_NAME)

    df.to_csv(out_path, index=False)
    print(f"\nĐã lưu → {out_path}")
    print(f"Shape  : {df.shape}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()