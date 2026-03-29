# src/model.py
"""
Model inference layer.
Hiện tại: mock bằng công thức đơn giản.
Sau này: load pkl và gọi model.predict() là xong.
"""

import random
import joblib
import os
from typing import TypedDict

MODEL_PATH = "models/pm25_model.pkl"


# ─── Types ───────────────────────────────────────────────────────────────────

class Features(TypedDict):
    pm25_lag1:   float
    pm25_lag3:   float
    pm25_lag24:  float
    temperature: float
    humidity:    int
    wind_speed:  float


# ─── Mock predict ────────────────────────────────────────────────────────────

def _mock_predict(features: Features, horizon: int) -> list[float]:
    """
    Giả lập dự báo bằng công thức đơn giản + noise.
    Kết quả trông 'hợp lý' để test UI.
    """
    base = (
        features["pm25_lag1"] * 0.5
        + features["pm25_lag3"] * 0.3
        + features["pm25_lag24"] * 0.2
    )

    # Humidity cao → PM2.5 tăng nhẹ
    base += (features["humidity"] - 70) * 0.1

    # Gió mạnh → PM2.5 giảm
    base -= features["wind_speed"] * 1.5

    base = max(5.0, base)

    results = []
    val = base
    for _ in range(horizon):
        val += random.gauss(0, 2.5)   # noise nhỏ mỗi bước
        val = max(5.0, min(300.0, val))
        results.append(round(val, 1))

    return results


# ─── Public API ──────────────────────────────────────────────────────────────

def predict_multi_horizon(features: Features, horizon: int) -> list[float]:
    """
    Predict PM2.5 cho t+1 đến t+horizon.
    Trả về list[float] độ dài = horizon.

    Tự động fallback về mock nếu chưa có model file.
    """
    if not os.path.exists(MODEL_PATH):
        # ── Chưa có model → dùng mock ──────────────────────────────────────
        return _mock_predict(features, horizon)

    # ── Có model → load và predict ─────────────────────────────────────────
    # TODO: uncomment khi đã có model thực
    #
    # model = joblib.load(MODEL_PATH)
    # import numpy as np
    # X = np.array([[
    #     features["pm25_lag1"],
    #     features["pm25_lag3"],
    #     features["pm25_lag24"],
    #     features["temperature"],
    #     features["humidity"],
    #     features["wind_speed"],
    # ]])
    # predictions = []
    # for h in range(1, horizon + 1):
    #     X[0][0] = features[f"pm25_lag{h}"] if h <= 3 else predictions[-1]
    #     predictions.append(float(model.predict(X)[0]))
    # return predictions

    return _mock_predict(features, horizon)  # xóa dòng này khi uncomment trên