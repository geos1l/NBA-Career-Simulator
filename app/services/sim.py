from __future__ import annotations

import math
import random
import time
from statistics import median
from typing import Any, Dict, Iterator, List

import numpy as np

from app.services.external import get_career_stats, get_player_bio
from app.services.model_store import LoadedModel


TARGET_BOUNDS = {
    "next_gp": (20.0, 82.0),
    "next_min": (8.0, 40.0),
    "next_pts": (0.5, 45.0),
    "next_reb": (0.5, 20.0),
    "next_ast": (0.2, 14.0),
    "next_stl": (0.1, 4.0),
    "next_blk": (0.0, 5.0),
    "next_tov": (0.1, 7.0),
    "next_fg_pct": (0.30, 0.70),
    "next_fg3_pct": (0.18, 0.52),
    "next_ft_pct": (0.45, 0.96),
}


def _clip(metric: str, value: float) -> float:
    lo, hi = TARGET_BOUNDS[metric]
    return max(lo, min(hi, value))


# Pre-compute bounds arrays for vectorized clipping (column order = target_columns).
def _build_bounds_arrays(target_columns: List[str]):
    lows = np.array([TARGET_BOUNDS[c][0] for c in target_columns])
    highs = np.array([TARGET_BOUNDS[c][1] for c in target_columns])
    return lows, highs


def _aging_multiplier(age: int) -> float:
    # Age curve: slight boost through peak years, decline after 32.
    if age <= 24:
        return 1.04
    if age <= 27:
        return 1.02
    if age <= 30:
        return 1.00
    if age <= 32:
        return 0.99
    if age <= 35:
        return 0.96
    if age <= 38:
        return 0.92
    return 0.86


def _retirement_risk(age: int, mpg: float, ppg: float, career_seasons: int) -> float:
    # Dynamic retirement benchmark with age, role and longevity.
    age_term = 1.0 / (1.0 + math.exp(-(age - 35) / 1.8))
    role_term = 1.0 / (1.0 + math.exp((mpg - 16) / 3.0))
    production_term = 1.0 / (1.0 + math.exp((ppg - 8) / 2.0))
    tenure_term = max(0.0, (career_seasons - 14) * 0.015)
    risk = 0.008 + 0.40 * age_term + 0.20 * role_term + 0.16 * production_term + tenure_term
    return max(0.0, min(0.92, risk))



def _compute_derived(pred: Dict[str, float]) -> Dict[str, float]:
    gp = max(pred["next_gp"], 1.0)
    ppg = pred["next_pts"]
    rpg = pred["next_reb"]
    apg = pred["next_ast"]
    spg = pred["next_stl"]
    bpg = pred["next_blk"]
    tpg = pred["next_tov"]
    mpg = pred["next_min"]

    points_total = round(ppg * gp, 1)
    reb_total = round(rpg * gp, 1)
    ast_total = round(apg * gp, 1)
    stl_total = round(spg * gp, 1)
    blk_total = round(bpg * gp, 1)
    tov_total = round(tpg * gp, 1)
    min_total = round(mpg * gp, 1)

    # Approximate makes/attempts for derived metrics.
    fga_pg = max(5.0, ppg * 0.9)
    fgm_pg = fga_pg * pred["next_fg_pct"]
    fg3a_pg = max(0.5, fga_pg * 0.38)
    fg3m_pg = fg3a_pg * pred["next_fg3_pct"]
    fta_pg = max(1.0, ppg * 0.28)
    ftm_pg = fta_pg * pred["next_ft_pct"]

    fga = fga_pg * gp
    fgm = fgm_pg * gp
    fg3m = fg3m_pg * gp
    fta = fta_pg * gp
    ftm = ftm_pg * gp
    efg_pct = (fgm + 0.5 * fg3m) / fga if fga > 0 else 0.0
    ts_den = 2.0 * (fga + 0.44 * fta)
    ts_pct = points_total / ts_den if ts_den > 0 else 0.0
    ast_to_tov = ast_total / tov_total if tov_total > 0 else 0.0
    efficiency = (
        points_total
        + reb_total
        + ast_total
        + stl_total
        + blk_total
        - (fga - fgm)
        - (fta - ftm)
        - tov_total
    )

    return {
        "per_game": {
            "ppg": round(ppg, 1),
            "rpg": round(rpg, 1),
            "apg": round(apg, 1),
            "spg": round(spg, 1),
            "bpg": round(bpg, 1),
            "tpg": round(tpg, 1),
            "mpg": round(mpg, 1),
            "fg_pct": round(pred["next_fg_pct"], 3),
            "fg3_pct": round(pred["next_fg3_pct"], 3),
            "ft_pct": round(pred["next_ft_pct"], 3),
        },
        "totals": {
            "pts": points_total,
            "reb": reb_total,
            "ast": ast_total,
            "stl": stl_total,
            "blk": blk_total,
            "tov": tov_total,
            "min": min_total,
        },
        "advanced": {
            "efg_pct": round(efg_pct, 3),
            "ts_pct": round(ts_pct, 3),
            "ast_to_tov": round(ast_to_tov, 2),
            "efficiency": round(efficiency, 1),
        },
    }


def _to_feature_vector(row: Dict[str, Any], age: int) -> List[float]:
    pg = row["per_game"]
    return [
        float(age),
        float(row["gp"]),
        float(pg["mpg"]),
        float(pg["ppg"]),
        float(pg["rpg"]),
        float(pg["apg"]),
        float(pg["spg"]),
        float(pg["bpg"]),
        float(pg["tpg"]),
        float(pg["fg_pct"]),
        float(pg["fg3_pct"]),
        float(pg["ft_pct"]),
    ]


def simulate_player_events(
    loaded_model: LoadedModel,
    player_id: int,
    start_season: int,
    simulations: int = 250,
) -> Iterator[Dict[str, Any]]:
    """
    Vectorized Monte Carlo simulation.

    Instead of running N paths sequentially (N × Y individual predict calls),
    this batches all N paths into a single predict call per year offset
    (Y batch calls total).  For 250 paths × 17 years this reduces ~4,250
    sklearn calls to ~17 — typically 20-50× faster.

    Yields real progress events per year offset so the loading bar reflects
    actual computation, not a fake timer.
    """
    artifact = loaded_model.artifact
    model = artifact["model"]
    target_columns: List[str] = artifact["target_columns"]
    residual_std: List[float] = artifact["residual_std"]
    n_targets = len(target_columns)
    std_arr = np.maximum(0.01, np.array(residual_std, dtype=float))
    bounds_lo, bounds_hi = _build_bounds_arrays(target_columns)

    # Column index lookups for vectorized aging.
    col_idx = {c: i for i, c in enumerate(target_columns)}
    aging_cols = [col_idx[c] for c in ("next_pts", "next_reb", "next_ast") if c in col_idx]

    yield {
        "event": "progress",
        "pct": 1,
        "phase": "career",
        "message": "Loading player career…",
    }
    career = get_career_stats(player_id)
    if not career:
        raise ValueError("No career data found for this player.")

    matching = [row for row in career if row["season_start"] == start_season]
    if not matching:
        raise ValueError("Start season not found in player's career.")

    start_index = career.index(matching[0])
    if start_index < 2:
        raise ValueError("Simulation requires at least 3 seasons of data before the start point.")

    base = career[start_index]
    base_age = int(base["age"] or 20)
    prior_years = start_index + 1
    remaining_cap = max(1, 20 - prior_years)
    noise_scale = 1.0

    N = simulations

    yield {
        "event": "progress",
        "pct": 5,
        "phase": "paths",
        "done": 0,
        "total": remaining_cap,
        "eta_seconds": None,
        "message": f"Simulating {N} paths × {remaining_cap} years…",
    }

    # ── Vectorized state: one row per path ──
    base_vec = _to_feature_vector(base, base_age)
    # current_features[i] = [age, gp, mpg, ppg, rpg, apg, spg, bpg, tpg, fg_pct, fg3_pct, ft_pct]
    current_features = np.tile(base_vec, (N, 1))  # (N, 12)

    alive = np.ones(N, dtype=bool)  # tracks which paths haven't retired
    # Store per-path results as list of lists (only alive paths get appended).
    paths: List[List[Dict[str, Any]]] = [[] for _ in range(N)]

    t_loop = time.perf_counter()

    for year_i in range(1, remaining_cap + 1):
        age = base_age + year_i
        n_alive = int(alive.sum())
        if n_alive == 0:
            break

        alive_idx = np.where(alive)[0]
        feats = current_features[alive_idx].copy()
        feats[:, 0] = float(age)  # update age column

        # ── Single batch predict for all alive paths ──
        preds = model.predict(feats)  # (n_alive, n_targets)

        # ── Vectorized noise injection ──
        noise = np.random.normal(0, std_arr * noise_scale, size=(n_alive, n_targets))
        draws = preds + noise

        # Aging multiplier on pts/reb/ast columns.
        aging_mult = _aging_multiplier(age)
        for ci in aging_cols:
            draws[:, ci] *= aging_mult

        # Clip to bounds.
        draws = np.clip(draws, bounds_lo, bounds_hi)

        # ── Build per-path season dicts and update state ──
        for k, ai in enumerate(alive_idx):
            predicted = {col: float(draws[k, j]) for j, col in enumerate(target_columns)}
            derived = _compute_derived(predicted)
            season_proj = {
                "season_offset": year_i,
                "age": age,
                "gp": int(round(predicted["next_gp"])),
                **derived,
            }
            paths[ai].append(season_proj)

            # Update current features for this path's next iteration.
            pg = season_proj["per_game"]
            current_features[ai] = [
                float(age), float(season_proj["gp"]),
                float(pg["mpg"]), float(pg["ppg"]), float(pg["rpg"]),
                float(pg["apg"]), float(pg["spg"]), float(pg["bpg"]),
                float(pg["tpg"]), float(pg["fg_pct"]), float(pg["fg3_pct"]),
                float(pg["ft_pct"]),
            ]
            # Retirement check.
            if age >= 31:
                risk = _retirement_risk(
                    age=age,
                    mpg=pg["mpg"],
                    ppg=pg["ppg"],
                    career_seasons=prior_years + year_i,
                )
                if random.random() < risk:
                    alive[ai] = False

        # ── Progress event per year offset ──
        elapsed = time.perf_counter() - t_loop
        per_year = elapsed / year_i
        eta = per_year * (remaining_cap - year_i)
        pct = 5 + int(87 * year_i / remaining_cap)
        yield {
            "event": "progress",
            "pct": pct,
            "phase": "paths",
            "done": year_i,
            "total": remaining_cap,
            "eta_seconds": round(eta, 1),
            "message": f"Year {year_i}/{remaining_cap} ({int(alive.sum())} paths alive)",
        }

    # ── Aggregation ──
    max_offset = max((len(p) for p in paths), default=0)
    aggregated: List[Dict[str, Any]] = []

    def _pct(values: List[float], q: float) -> float:
        return round(float(np.quantile(np.array(values, dtype=float), q)), 1)

    for offset in range(1, max_offset + 1):
        bucket = [p[offset - 1] for p in paths if len(p) >= offset]
        if not bucket:
            continue

        ppg_vals = [b["per_game"]["ppg"] for b in bucket]
        rpg_vals = [b["per_game"]["rpg"] for b in bucket]
        apg_vals = [b["per_game"]["apg"] for b in bucket]
        mpg_vals = [b["per_game"]["mpg"] for b in bucket]

        aggregated.append(
            {
                "season_offset": offset,
                "age_median": int(round(median([b["age"] for b in bucket]))),
                "gp_median": int(round(median([b["gp"] for b in bucket]))),
                "metrics": {
                    "ppg": {"p10": _pct(ppg_vals, 0.10), "p50": _pct(ppg_vals, 0.50), "p90": _pct(ppg_vals, 0.90)},
                    "rpg": {"p10": _pct(rpg_vals, 0.10), "p50": _pct(rpg_vals, 0.50), "p90": _pct(rpg_vals, 0.90)},
                    "apg": {"p10": _pct(apg_vals, 0.10), "p50": _pct(apg_vals, 0.50), "p90": _pct(apg_vals, 0.90)},
                    "mpg": {"p10": _pct(mpg_vals, 0.10), "p50": _pct(mpg_vals, 0.50), "p90": _pct(mpg_vals, 0.90)},
                },
                "sample_projection": bucket[0],
            }
        )

    projected_retire_ages = [p[-1]["age"] for p in paths if p]
    projected_retire_age = int(round(median(projected_retire_ages))) if projected_retire_ages else base_age

    yield {
        "event": "progress",
        "pct": 98,
        "phase": "bio",
        "message": "Loading player profile…",
    }
    bio = get_player_bio(player_id)
    result = {
        "player_id": player_id,
        "player_name": bio.get("name"),
        "position": bio.get("position"),
        "start_season": start_season,
        "base_age": base_age,
        "simulations": simulations,
        "projected_retirement_age": projected_retire_age,
        "model_version": loaded_model.version,
        "paths_sample": max(paths, key=len) if paths else [],
        "aggregated_projection": aggregated,
    }
    yield {"event": "complete", "pct": 100, "result": result}


def simulate_player(
    loaded_model: LoadedModel,
    player_id: int,
    start_season: int,
    simulations: int = 250,
) -> Dict[str, Any]:
    for msg in simulate_player_events(
        loaded_model,
        player_id,
        start_season,
        simulations=simulations,
    ):
        if msg.get("event") == "complete":
            return msg["result"]
    raise RuntimeError("Simulation finished without a result")