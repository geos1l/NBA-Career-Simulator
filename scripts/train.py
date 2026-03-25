"""Train the NBA career simulator model with CLI-configurable hyperparameters.

Usage:
    python scripts/train.py                          # defaults, saves as next version
    python scripts/train.py --version v2             # explicit version label
    python scripts/train.py --learning-rate 0.03 --max-depth 8 --max-iter 500
    python scripts/train.py --set-active             # make the new model the active one

Prints a comparison table (MAE / RMSE / R² per stat) against the current
active model so you can decide whether to promote the new version.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.multioutput import MultiOutputRegressor  # noqa: E402

from app.services.model_store import ModelStore  # noqa: E402
from app.services.trainer import (  # noqa: E402
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    _build_training_rows,
)

STAT_LABELS = {
    "next_gp": "GP",
    "next_min": "MIN",
    "next_pts": "PTS",
    "next_reb": "REB",
    "next_ast": "AST",
    "next_stl": "STL",
    "next_blk": "BLK",
    "next_tov": "TOV",
    "next_fg_pct": "FG%",
    "next_fg3_pct": "FG3%",
    "next_ft_pct": "FT%",
}


def print_comparison_table(
    target_cols: list[str],
    new_metrics: dict,
    old_metrics: dict | None,
) -> None:
    """Print a formatted comparison table of MAE / RMSE / R² per stat."""
    header = f"{'Stat':<6} {'MAE':>7} {'RMSE':>7} {'R²':>7}"
    if old_metrics:
        header += f"  │ {'Old MAE':>7} {'Δ MAE':>7}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for col in target_cols:
        label = STAT_LABELS.get(col, col)
        mae = new_metrics["mae"][col]
        rmse = new_metrics["rmse"][col]
        r2 = new_metrics["r2"][col]
        row = f"{label:<6} {mae:>7.3f} {rmse:>7.3f} {r2:>7.3f}"
        if old_metrics and col in old_metrics.get("mae", {}):
            old_mae = old_metrics["mae"][col]
            delta = mae - old_mae
            arrow = "▼" if delta < -0.001 else ("▲" if delta > 0.001 else "═")
            row += f"  │ {old_mae:>7.3f} {arrow}{abs(delta):>6.3f}"
        print(row)

    # Averages
    avg_mae = np.mean([new_metrics["mae"][c] for c in target_cols])
    avg_rmse = np.mean([new_metrics["rmse"][c] for c in target_cols])
    avg_r2 = np.mean([new_metrics["r2"][c] for c in target_cols])
    print("-" * len(header))
    avg_row = f"{'AVG':<6} {avg_mae:>7.3f} {avg_rmse:>7.3f} {avg_r2:>7.3f}"
    if old_metrics:
        old_avg = np.mean([old_metrics["mae"].get(c, 0) for c in target_cols])
        delta = avg_mae - old_avg
        arrow = "▼" if delta < -0.001 else ("▲" if delta > 0.001 else "═")
        avg_row += f"  │ {old_avg:>7.3f} {arrow}{abs(delta):>6.3f}"
    print(avg_row)
    print("=" * len(header))
    print("▼ = improved, ▲ = worse, ═ = same")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NBA career simulator model.")
    parser.add_argument("--start-year", type=int, default=2000,
                        help="First season year in training data (default: 2000)")
    parser.add_argument("--end-year", type=int, default=None,
                        help="Last season year (default: current year - 2)")
    parser.add_argument("--version", type=str, default=None,
                        help="Version label (default: auto-generated timestamp)")
    parser.add_argument("--set-active", action="store_true",
                        help="Set the new model as active (updates latest.json)")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--max-iter", type=int, default=350)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Validation split ratio (default: 0.2)")
    args = parser.parse_args()

    now_year = datetime.now().year
    end_year = args.end_year if args.end_year is not None else (now_year - 2)
    if end_year < args.start_year:
        parser.error("end-year must be >= start-year")

    version = args.version or datetime.now(tz=timezone.utc).strftime("v%Y%m%dT%H%M%SZ")
    store = ModelStore()

    # ── Load current active model metrics for comparison ──
    old_metrics = None
    try:
        current = store.load()
        old_mae = current.metadata.get("validation_mae", {})
        if old_mae:
            old_metrics = {"mae": old_mae}
            print(f"Current active model: {current.version}")
    except FileNotFoundError:
        print("No existing model found — training from scratch.")

    # ── Build training data ──
    print(f"\nFetching training data: seasons {args.start_year}–{end_year}...")
    t0 = time.perf_counter()
    frame = _build_training_rows(start_year=args.start_year, end_year=end_year)
    if frame.empty:
        print("ERROR: No training data. Check nba_api network access.")
        sys.exit(1)
    t_data = time.perf_counter() - t0
    print(f"  {len(frame)} training rows in {t_data:.1f}s")

    X = frame[FEATURE_COLUMNS].to_numpy()
    y = frame[TARGET_COLUMNS].to_numpy()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    # ── Train ──
    print(f"\nTraining: lr={args.learning_rate}, depth={args.max_depth}, "
          f"iter={args.max_iter}, min_leaf={args.min_samples_leaf}")
    t0 = time.perf_counter()
    model = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            max_iter=args.max_iter,
            min_samples_leaf=args.min_samples_leaf,
            random_state=42,
        )
    )
    model.fit(X_train, y_train)
    t_train = time.perf_counter() - t0
    print(f"  Trained in {t_train:.1f}s")

    # ── Evaluate ──
    y_pred = model.predict(X_val)
    residuals = y_val - y_pred
    residual_std = np.std(residuals, axis=0).tolist()

    new_metrics = {"mae": {}, "rmse": {}, "r2": {}}
    for i, col in enumerate(TARGET_COLUMNS):
        new_metrics["mae"][col] = float(mean_absolute_error(y_val[:, i], y_pred[:, i]))
        new_metrics["rmse"][col] = float(np.sqrt(mean_squared_error(y_val[:, i], y_pred[:, i])))
        new_metrics["r2"][col] = float(r2_score(y_val[:, i], y_pred[:, i]))

    print_comparison_table(TARGET_COLUMNS, new_metrics, old_metrics)

    # ── Save ──
    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "residual_std": residual_std,
    }
    metadata = {
        "version": version,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "train_start_year": args.start_year,
        "train_end_year": end_year,
        "rows": len(frame),
        "features": FEATURE_COLUMNS,
        "targets": TARGET_COLUMNS,
        "validation_mae": new_metrics["mae"],
        "validation_rmse": new_metrics["rmse"],
        "validation_r2": new_metrics["r2"],
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "max_iter": args.max_iter,
            "min_samples_leaf": args.min_samples_leaf,
            "test_size": args.test_size,
        },
    }
    store.save(artifact, metadata, set_active=args.set_active)
    print(f"\nSaved model as '{version}'")

    if args.set_active:
        print(f"Set as active model (latest.json -> {version})")
    elif old_metrics:
        print("NOT set as active. Run again with --set-active to promote.")

    print("Done.")


if __name__ == "__main__":
    main()
