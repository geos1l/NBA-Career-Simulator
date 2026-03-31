from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from app.database import DB_PATH
from app.services.model_store import ModelStore


FEATURE_COLUMNS = [
    "age",
    "gp",
    "min",
    "pts",
    "reb",
    "ast",
    "stl",
    "blk",
    "tov",
    "fg_pct",
    "fg3_pct",
    "ft_pct",
]

TARGET_COLUMNS = [
    "next_gp",
    "next_min",
    "next_pts",
    "next_reb",
    "next_ast",
    "next_stl",
    "next_blk",
    "next_tov",
    "next_fg_pct",
    "next_fg3_pct",
    "next_ft_pct",
]


@dataclass
class TrainOutput:
    artifact: Dict[str, Any]
    metadata: Dict[str, Any]


def _build_training_rows(start_year: int, end_year: int, min_gp: int = 20) -> pd.DataFrame:
    import sqlite3

    conn = sqlite3.connect(str(DB_PATH))
    try:
        query = """
            SELECT
                c.player_id,
                c.season_start,
                c.age,
                c.gp,
                c.mpg   AS min,
                c.ppg   AS pts,
                c.rpg   AS reb,
                c.apg   AS ast,
                c.spg   AS stl,
                c.bpg   AS blk,
                c.tpg   AS tov,
                c.fg_pct,
                c.fg3_pct,
                c.ft_pct,
                n.gp    AS next_gp,
                n.mpg   AS next_min,
                n.ppg   AS next_pts,
                n.rpg   AS next_reb,
                n.apg   AS next_ast,
                n.spg   AS next_stl,
                n.bpg   AS next_blk,
                n.tpg   AS next_tov,
                n.fg_pct  AS next_fg_pct,
                n.fg3_pct AS next_fg3_pct,
                n.ft_pct  AS next_ft_pct
            FROM season_stats c
            JOIN season_stats n
              ON  c.player_id = n.player_id
              AND n.season_start = c.season_start + 1
            WHERE c.season_start >= ?
              AND c.season_start <= ?
              AND c.gp >= ?
              AND n.gp >= ?
        """
        frame = pd.read_sql_query(query, conn, params=(start_year, end_year, min_gp, min_gp))
    finally:
        conn.close()

    return frame


def train_model(start_year: int = 2000, end_year: int | None = None) -> TrainOutput:
    now_year = datetime.now().year
    train_end = end_year if end_year is not None else (now_year - 2)
    frame = _build_training_rows(start_year=start_year, end_year=train_end)
    if frame.empty:
        raise RuntimeError("Training data is empty. Run scripts/ingest_data.py first.")

    X = frame[FEATURE_COLUMNS].to_numpy()
    y = frame[TARGET_COLUMNS].to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=6,
            max_iter=350,
            min_samples_leaf=20,
            random_state=42,
        )
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    residuals = y_val - y_pred
    residual_std = np.std(residuals, axis=0).tolist()
    metric_mae = {
        TARGET_COLUMNS[i]: float(mean_absolute_error(y_val[:, i], y_pred[:, i])) for i in range(len(TARGET_COLUMNS))
    }

    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "residual_std": residual_std,
    }
    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "train_start_year": start_year,
        "train_end_year": train_end,
        "rows": int(len(frame)),
        "features": FEATURE_COLUMNS,
        "targets": TARGET_COLUMNS,
        "validation_mae": metric_mae,
    }
    return TrainOutput(artifact=artifact, metadata=metadata)


def ensure_trained_model(store: ModelStore) -> Dict[str, Any]:
    """Load existing model or train+save (for scripts/CLI only — the API uses load-only paths)."""
    try:
        loaded = store.load()
        return {"status": "loaded", "version": loaded.version, "metadata": loaded.metadata}
    except FileNotFoundError:
        output = train_model()
        metadata = store.save(output.artifact, output.metadata)
        return {"status": "trained", "version": metadata["version"], "metadata": metadata}

