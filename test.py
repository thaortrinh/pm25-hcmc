"""
Data Quality Check for PM2.5 Raw Data
=====================================
Kiểm tra missing, time index integrity, outlier, duplicate.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# ─── Setup ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
PM25_FILE = DATA_RAW / "pm25_sensor_11357424.csv"

print("=" * 80)
print("DATA QUALITY CHECK — PM2.5 Raw Data")
print("=" * 80)

# ─── Load ────────────────────────────────────────────────────────────────────

try:
    df = pd.read_csv(PM25_FILE)
    print(f"\n✓ Loaded: {PM25_FILE.name}")
    print(f"  Shape: {df.shape}")
except Exception as e:
    print(f"✗ Error loading file: {e}")
    exit(1)

# ─── 1. Basic Info ────────────────────────────────────────────────────────────

print("\n" + "─" * 80)
print("1. BASIC INFO")
print("─" * 80)

# Normalize column names
df.columns = df.columns.str.lower()
print(f"\nColumns: {list(df.columns)}")
print(f"Data types:\n{df.dtypes}")
print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

# ─── 2. Datetime Parsing ──────────────────────────────────────────────────────

print("\n" + "─" * 80)
print("2. DATETIME PARSING & CONVERSION")
print("─" * 80)

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

# Check for parsing errors
n_dt_error = df["datetime"].isna().sum()
if n_dt_error > 0:
    print(f"\n✗ {n_dt_error} rows have invalid datetime")
    print(f"  (Invalid rows will be dropped for further analysis)")
    df = df.dropna(subset=["datetime"])
else:
    print(f"\n✓ All {len(df)} rows parsed successfully")

print(f"  Date range: {df['datetime'].min()} → {df['datetime'].max()}")
print(f"  Duration: {(df['datetime'].max() - df['datetime'].min()).days} days")

# ─── 3. Time Index Integrity ──────────────────────────────────────────────────

print("\n" + "─" * 80)
print("3. TIME INDEX INTEGRITY")
print("─" * 80)

# Sort by datetime
df_sorted = df.sort_values("datetime").reset_index(drop=True)
is_sorted = (df_sorted["datetime"] == df.sort_values("datetime")["datetime"]).all()
print(f"\n{'✓' if is_sorted else '✗'} Data is {'already' if is_sorted else 'NOT'} sorted by datetime")

# Check for duplicates
n_dup = df_sorted.duplicated(subset=["datetime"]).sum()
if n_dup > 0:
    print(f"\n✗ {n_dup} duplicate timestamps found:")
    dups = df_sorted[df_sorted.duplicated(subset=["datetime"], keep=False)].sort_values("datetime")
    print(dups[["datetime", "pm25_avg"]].head(10))
else:
    print(f"\n✓ No duplicate timestamps")

# Reindex to full hourly range and check gaps
dt_min = df_sorted["datetime"].min()
dt_max = df_sorted["datetime"].max()
n_hours_expected = int((dt_max - dt_min).total_seconds() / 3600) + 1
n_hours_actual = len(df_sorted)

print(f"\n  Expected hourly records: {n_hours_expected:,}")
print(f"  Actual records: {n_hours_actual:,}")
print(f"  Missing records: {n_hours_expected - n_hours_actual:,} ({(n_hours_expected - n_hours_actual) / n_hours_expected * 100:.2f}%)")

# Show gap pattern
full_index = pd.date_range(dt_min, dt_max, freq="h")
df_reindex = df_sorted.set_index("datetime").reindex(full_index)
gaps = df_reindex["pm25_avg"].isna()
n_gaps = gaps.sum()

if n_gaps > 0:
    print(f"\n  Gap locations (missing hours by month-hour):")
    gap_idx = df_reindex.index[gaps]
    gap_summary = pd.DataFrame({
    "datetime": gap_idx,
    "year":  gap_idx.year,
    "month": gap_idx.month,
    "hour":  gap_idx.hour,
    })
    gap_pivot = gap_summary.groupby(["year", "month", "hour"]).size().unstack(fill_value=0)
    print(gap_pivot)

# ─── 4. Missing Data Pattern ──────────────────────────────────────────────────

print("\n" + "─" * 80)
print("4. MISSING DATA PATTERN")
print("─" * 80)

missing_pct = df_sorted.isna().mean() * 100
print(f"\nMissing % by column:")
for col in df_sorted.columns:
    pct = missing_pct[col]
    status = "✓" if pct == 0 else "⚠" if pct < 10 else "✗"
    print(f"  {status} {col:20s}: {pct:6.2f}%")

# ─── 5. Outlier & Value Range ─────────────────────────────────────────────────

print("\n" + "─" * 80)
print("5. OUTLIER & VALUE RANGE")
print("─" * 80)

for col in ["pm25_avg", "pm25_min", "pm25_max", "pm25_median", "coverage_pct"]:
    if col not in df_sorted.columns:
        continue
    
    valid = df_sorted[col].dropna()
    if len(valid) == 0:
        print(f"\n  {col}: NO DATA")
        continue
    
    print(f"\n  {col}:")
    print(f"    Count    : {len(valid):,}")
    print(f"    Min      : {valid.min():.2f}")
    print(f"    Q1       : {valid.quantile(0.25):.2f}")
    print(f"    Median   : {valid.median():.2f}")
    print(f"    Mean     : {valid.mean():.2f}")
    print(f"    Q3       : {valid.quantile(0.75):.2f}")
    print(f"    Max      : {valid.max():.2f}")
    print(f"    Std      : {valid.std():.2f}")
    
    # Flag suspicious values
    if col == "pm25_avg":
        n_neg = (valid < 0).sum()
        n_extreme = (valid > 500).sum()
        if n_neg > 0:
            print(f"    ✗ Negative values: {n_neg}")
        if n_extreme > 0:
            print(f"    ✗ Extreme (>500): {n_extreme}")
        if n_neg == 0 and n_extreme == 0:
            print(f"    ✓ No extreme values detected")

# ─── 6. Data Quality Score ───────────────────────────────────────────────────

print("\n" + "─" * 80)
print("6. DATA QUALITY SCORE")
print("─" * 80)

scores = {
    "Datetime parsing": 100 if n_dt_error == 0 else 50,
    "Time sorting": 100 if is_sorted else 0,
    "No duplicates": 100 if n_dup == 0 else 50,
    "Time continuity": 100 * (1 - n_gaps / n_hours_expected),
    "PM2.5 completeness": 100 * (1 - df_sorted["pm25_avg"].isna().sum() / len(df_sorted)),
    "PM2.5 validity": 100 if ((valid >= 0) & (valid <= 500)).all() else 70,
}

print("\nComponent scores:")
for component, score in scores.items():
    print(f"  {component:25s}: {score:5.1f}%")

overall_score = np.mean(list(scores.values()))
print(f"\n  OVERALL QUALITY SCORE: {overall_score:.1f}%")
if overall_score >= 90:
    status = "✓ EXCELLENT"
elif overall_score >= 75:
    status = "⚠ GOOD"
elif overall_score >= 50:
    status = "⚠ FAIR (needs attention)"
else:
    status = "✗ POOR (significant issues)"
print(f"  Status: {status}")

# ─── 7. Recommendations ──────────────────────────────────────────────────────

print("\n" + "─" * 80)
print("7. RECOMMENDATIONS")
print("─" * 80)

recommendations = []

if n_dt_error > 0:
    recommendations.append(f"• Fix {n_dt_error} datetime parsing errors")

if n_dup > 0:
    recommendations.append(f"• Remove or investigate {n_dup} duplicate timestamps")

if n_gaps > 0:
    recommendations.append(f"• Handle {n_gaps} missing hours (consider interpolation or forward-fill)")

if df_sorted["pm25_avg"].isna().sum() > 0:
    recommendations.append(f"• Handle {df_sorted['pm25_avg'].isna().sum()} missing PM2.5 values")

pm25_valid = df_sorted["pm25_avg"].dropna()
if (pm25_valid < 0).sum() > 0 or (pm25_valid > 500).sum() > 0:
    recommendations.append(f"• Investigate extreme PM2.5 values outside [0, 500]")

if df_sorted["coverage_pct"].notna().sum() > 0 and (df_sorted["coverage_pct"] < 50).sum() > 0:
    recommendations.append(f"• Investigate {(df_sorted['coverage_pct'] < 50).sum()} hours with coverage < 50%")

if len(recommendations) == 0:
    recommendations.append("✓ Data looks good; ready for preprocessing pipeline")

print()
for rec in recommendations:
    print(rec)

# ─── 8. Sample Data ──────────────────────────────────────────────────────────

print("\n" + "─" * 80)
print("8. SAMPLE DATA (First 10 & Last 10 rows)")
print("─" * 80)

print("\nFirst 10 rows:")
print(df_sorted[["datetime", "pm25_avg", "pm25_min", "pm25_max", "pm25_median", "coverage_pct"]].head(10).to_string(index=False))

print("\n\nLast 10 rows:")
print(df_sorted[["datetime", "pm25_avg", "pm25_min", "pm25_max", "pm25_median", "coverage_pct"]].tail(10).to_string(index=False))

print("\n" + "=" * 80)
print("END OF REPORT")
print("=" * 80)
