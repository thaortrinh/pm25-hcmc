from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor

from src.inference.feature_builder import (
    PROJECT_ROOT,
    PROCESSED_REFERENCE_PATH,
    build_training_frame,
    fit_preprocessor_metadata,
    load_local_training_sources,
    load_metadata,
    merge_pm_and_weather,
    save_metadata,
)


MODEL_DIR = PROJECT_ROOT / "model" / "multi_6h_weights"
MODEL_PATH = MODEL_DIR / "catboost_multi_horizon_deployable.cbm"
METADATA_PATH = MODEL_DIR / "deployment_metadata.json"


@dataclass(slots=True)
class LoadedArtifact:
    model_path: Path
    metadata_path: Path
    model: CatBoostRegressor
    metadata: object
    generated: bool


def _candidate_model_paths() -> list[Path]:
    patterns = ["*.cbm", "*.bin", "*.joblib", "*.pkl"]
    candidates: list[Path] = []
    for root in [PROJECT_ROOT / "model", PROJECT_ROOT / "models"]:
        if not root.exists():
            continue
        for pattern in patterns:
            candidates.extend(root.rglob(pattern))
    return sorted({candidate.resolve() for candidate in candidates if candidate.is_file()})


def discover_model_artifact() -> Path | None:
    if MODEL_PATH.exists():
        return MODEL_PATH
    for candidate in _candidate_model_paths():
        lowered = candidate.name.lower()
        if "catboost" in lowered and "6h" in lowered:
            return candidate
        if candidate.suffix == ".cbm":
            return candidate
    return None


def train_deployable_artifact(force: bool = False) -> tuple[Path, Path]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and METADATA_PATH.exists() and not force:
        return MODEL_PATH, METADATA_PATH

    pm_history, weather_history = load_local_training_sources()
    merged = merge_pm_and_weather(pm_history, weather_history)
    reference = pd.read_csv(PROCESSED_REFERENCE_PATH) if PROCESSED_REFERENCE_PATH.exists() else None

    metadata = fit_preprocessor_metadata(merged, reconstruction_reference=reference)
    training_frame = build_training_frame(merged, metadata)
    X = training_frame[metadata.feature_columns].copy()
    y = training_frame[metadata.target_columns].copy()

    for column in metadata.categorical_columns:
        X[column] = X[column].astype(str)

    n = len(X)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)
    X_train = X.iloc[:train_end].copy()
    y_train = y.iloc[:train_end].copy()
    X_valid = X.iloc[train_end:valid_end].copy()
    y_valid = y.iloc[train_end:valid_end].copy()

    model = CatBoostRegressor(
        loss_function="MultiRMSE",
        eval_metric="MultiRMSE",
        iterations=2500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5,
        random_strength=1,
        subsample=0.8,
        bootstrap_type="Bernoulli",
        od_type="Iter",
        od_wait=120,
        random_seed=42,
        verbose=100,
    )
    model.fit(
        X_train,
        y_train,
        cat_features=metadata.categorical_columns,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
    )

    model.save_model(MODEL_PATH.as_posix())
    save_metadata(metadata, METADATA_PATH)
    (MODEL_DIR / "deployment_note.txt").write_text(
        "\n".join(
            [
                "Deployable artifact generated because no serialized CatBoost model was found in the repository.",
                "The CatBoost direct multi-horizon setup from model/6h_pm.py was preserved.",
                "target_next_hour was excluded because it is a future-leaking helper column and cannot be used at inference time.",
                f"Generated at: {datetime.utcnow().isoformat()}Z",
            ]
        ),
        encoding="utf-8",
    )
    return MODEL_PATH, METADATA_PATH


def load_or_create_artifact(force_rebuild: bool = False) -> LoadedArtifact:
    model_path = discover_model_artifact()
    metadata_path = METADATA_PATH
    generated = False
    if force_rebuild or model_path is None or not metadata_path.exists():
        model_path, metadata_path = train_deployable_artifact(force=force_rebuild)
        generated = True

    model = CatBoostRegressor()
    model.load_model(model_path.as_posix())
    metadata = load_metadata(metadata_path)
    return LoadedArtifact(
        model_path=model_path,
        metadata_path=metadata_path,
        model=model,
        metadata=metadata,
        generated=generated,
    )
