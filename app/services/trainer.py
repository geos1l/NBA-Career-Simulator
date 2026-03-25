from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from app.services.external import get_league_season_per_game
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


def season_label(start_year: int) -> str:
    end_short = str((start_year + 1) % 100).zfill(2)
    return f"{start_year}-{end_short}"


NBA_API_TIMEOUT = 90

def _build_training_rows(start_year: int, end_year: int) -> pd.DataFrame:
    import time
    by_season: Dict[int, pd.DataFrame] = {}
    for year in range(start_year, end_year + 1):
        label = season_label(year)
        for attempt in range(2):
            try:
                frame = get_league_season_per_game(label, min_gp=20, timeout=NBA_API_TIMEOUT)
                break
            except Exception as e:
                if "timed out" in str(e).lower() and attempt == 0:
                    time.sleep(2)
                    continue
                raise
        if frame.empty:
            continue
        frame = frame.copy()
        frame["season_start"] = year
        by_season[year] = frame

    if not by_season:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    years = sorted(by_season.keys())
    for year in years[:-1]:
        current = by_season[year]
        nxt = by_season.get(year + 1)
        if nxt is None or nxt.empty:
            continue

        next_cols = ["PLAYER_ID", "GP", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG_PCT", "FG3_PCT", "FT_PCT"]
        merged = current.merge(
            nxt[next_cols],
            on="PLAYER_ID",
            suffixes=("", "_NEXT"),
        )
        for _, row in merged.iterrows():
            rows.append(
                {
                    "player_id": int(row["PLAYER_ID"]),
                    "season_start": int(year),
                    "age": float(row["AGE"]),
                    "gp": float(row["GP"]),
                    "min": float(row["MIN"]),
                    "pts": float(row["PTS"]),
                    "reb": float(row["REB"]),
                    "ast": float(row["AST"]),
                    "stl": float(row["STL"]),
                    "blk": float(row["BLK"]),
                    "tov": float(row["TOV"]),
                    "fg_pct": float(row["FG_PCT"]),
                    "fg3_pct": float(row["FG3_PCT"]),
                    "ft_pct": float(row["FT_PCT"]),
                    "next_gp": float(row["GP_NEXT"]),
                    "next_min": float(row["MIN_NEXT"]),
                    "next_pts": float(row["PTS_NEXT"]),
                    "next_reb": float(row["REB_NEXT"]),
                    "next_ast": float(row["AST_NEXT"]),
                    "next_stl": float(row["STL_NEXT"]),
                    "next_blk": float(row["BLK_NEXT"]),
                    "next_tov": float(row["TOV_NEXT"]),
                    "next_fg_pct": float(row["FG_PCT_NEXT"]),
                    "next_fg3_pct": float(row["FG3_PCT_NEXT"]),
                    "next_ft_pct": float(row["FT_PCT_NEXT"]),
                }
            )

    return pd.DataFrame(rows)


def train_model(start_year: int = 2000, end_year: int | None = None) -> TrainOutput:
    now_year = datetime.now().year
    train_end = end_year if end_year is not None else (now_year - 2)
    frame = _build_training_rows(start_year=start_year, end_year=train_end)
    if frame.empty:
        raise RuntimeError("Training data is empty. Check nba_api network access and date range.")

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

