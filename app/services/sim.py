from __future__ import annotations

import math
import random
from statistics import median
from typing import Any, Dict, List

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


def _aging_multiplier(age: int) -> float:
    # Slightly arcade-friendly age curve: more upside through peak years.
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


def _upside_index(base_ppg: float, base_apg: float, base_mpg: float, base_age: int) -> float:
    score = 0.0
    score += min(1.0, base_ppg / 30.0) * 0.55
    score += min(1.0, base_apg / 9.0) * 0.20
    score += min(1.0, base_mpg / 37.0) * 0.15
    if base_age <= 24:
        score += 0.15
    elif base_age <= 27:
        score += 0.08
    return max(0.0, min(1.0, score))


def _sample_year_shock(upside: float, prev_shock: float, age: int) -> float:
    # Introduce off-years and breakout years for less linear trajectories.
    breakout_prob = max(0.08, min(0.26, 0.12 + 0.12 * upside - 0.01 * max(0, age - 28)))
    off_year_prob = max(0.08, min(0.22, 0.14 - 0.03 * upside + 0.01 * max(0, age - 31)))

    roll = random.random()
    if roll < breakout_prob:
        shock = random.uniform(0.04, 0.16)
    elif roll < breakout_prob + off_year_prob:
        shock = random.uniform(-0.12, -0.03)
    else:
        shock = random.uniform(-0.025, 0.03)

    # Bounceback tendency after a clear off-year.
    if prev_shock <= -0.06:
        shock += random.uniform(0.015, 0.05)
    return shock


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


def simulate_player(
    loaded_model: LoadedModel,
    player_id: int,
    start_season: int,
    simulations: int = 250,
    realism_profile: str = "realistic",
) -> Dict[str, Any]:
    artifact = loaded_model.artifact
    model = artifact["model"]
    target_columns: List[str] = artifact["target_columns"]
    residual_std: List[float] = artifact["residual_std"]

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
    noise_scale = 0.80 if realism_profile == "realistic" else 1.15
    base_upside = _upside_index(
        base_ppg=float(base["per_game"]["ppg"]),
        base_apg=float(base["per_game"]["apg"]),
        base_mpg=float(base["per_game"]["mpg"]),
        base_age=base_age,
    )

    paths: List[List[Dict[str, Any]]] = []
    for _ in range(simulations):
        current = {
            "gp": float(base["gp"]),
            "per_game": dict(base["per_game"]),
        }
        path: List[Dict[str, Any]] = []
        age = base_age
        prev_shock = 0.0

        for i in range(1, remaining_cap + 1):
            age += 1
            features = np.array([_to_feature_vector(current, age)], dtype=float)
            pred = model.predict(features)[0]

            predicted: Dict[str, float] = {}
            for j, col in enumerate(target_columns):
                std = max(0.01, residual_std[j])
                draw = float(np.random.normal(loc=pred[j], scale=std * noise_scale))
                if col in {"next_pts", "next_reb", "next_ast"}:
                    draw *= _aging_multiplier(age)
                predicted[col] = _clip(col, draw)

            shock = _sample_year_shock(base_upside, prev_shock, age)
            production_multiplier = 1.0 + shock
            playmaking_multiplier = 1.0 + (shock * 0.8)
            minutes_multiplier = 1.0 + (shock * 0.55)
            efficiency_multiplier = 1.0 + (shock * 0.35)

            predicted["next_pts"] = _clip("next_pts", predicted["next_pts"] * production_multiplier)
            predicted["next_reb"] = _clip("next_reb", predicted["next_reb"] * (1.0 + shock * 0.45))
            predicted["next_ast"] = _clip("next_ast", predicted["next_ast"] * playmaking_multiplier)
            predicted["next_min"] = _clip("next_min", predicted["next_min"] * minutes_multiplier)
            predicted["next_fg_pct"] = _clip("next_fg_pct", predicted["next_fg_pct"] * efficiency_multiplier)
            predicted["next_fg3_pct"] = _clip("next_fg3_pct", predicted["next_fg3_pct"] * efficiency_multiplier)

            # Preserve upside for young stars so early injury-era starts have plausible peaks.
            if age <= 27 and base_upside > 0.75:
                floor_ppg = current["per_game"]["ppg"] * 0.93
                predicted["next_pts"] = max(predicted["next_pts"], floor_ppg)

            derived = _compute_derived(predicted)
            season_proj = {
                "season_offset": i,
                "age": age,
                "gp": int(round(predicted["next_gp"])),
                **derived,
            }
            path.append(season_proj)

            current = {
                "gp": season_proj["gp"],
                "per_game": season_proj["per_game"],
            }
            prev_shock = shock

            risk = _retirement_risk(
                age=age,
                mpg=season_proj["per_game"]["mpg"],
                ppg=season_proj["per_game"]["ppg"],
                career_seasons=prior_years + i,
            )
            if random.random() < risk and age >= 31:
                break

        paths.append(path)

    max_offset = max((len(p) for p in paths), default=0)
    aggregated: List[Dict[str, Any]] = []
    for offset in range(1, max_offset + 1):
        bucket = [p[offset - 1] for p in paths if len(p) >= offset]
        if not bucket:
            continue

        ages = [b["age"] for b in bucket]
        gp_vals = [b["gp"] for b in bucket]
        ppg_vals = [b["per_game"]["ppg"] for b in bucket]
        rpg_vals = [b["per_game"]["rpg"] for b in bucket]
        apg_vals = [b["per_game"]["apg"] for b in bucket]
        mpg_vals = [b["per_game"]["mpg"] for b in bucket]

        def _pct(values: List[float], q: float) -> float:
            return round(float(np.quantile(np.array(values, dtype=float), q)), 1)

        aggregated.append(
            {
                "season_offset": offset,
                "age_median": int(round(median(ages))),
                "gp_median": int(round(median(gp_vals))),
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
    bio = get_player_bio(player_id)
    return {
        "player_id": player_id,
        "player_name": bio.get("name"),
        "position": bio.get("position"),
        "start_season": start_season,
        "base_age": base_age,
        "simulations": simulations,
        "projected_retirement_age": projected_retire_age,
        "model_version": loaded_model.version,
        "paths_sample": paths[0] if paths else [],
        "aggregated_projection": aggregated,
    }