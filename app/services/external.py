from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from nba_api.stats.endpoints import commonplayerinfo, leaguedashplayerstats, playercareerstats
from nba_api.stats.static import players


PER_GAME_COLUMNS = {
    "ppg": "PTS",
    "rpg": "REB",
    "apg": "AST",
    "spg": "STL",
    "bpg": "BLK",
    "tpg": "TOV",
    "mpg": "MIN",
    "fg_pct": "FG_PCT",
    "fg3_pct": "FG3_PCT",
    "ft_pct": "FT_PCT",
}

TOTAL_COLUMNS = {
    "pts": "PTS",
    "reb": "REB",
    "ast": "AST",
    "stl": "STL",
    "blk": "BLK",
    "tov": "TOV",
    "min": "MIN",
    "fgm": "FGM",
    "fga": "FGA",
    "fg3m": "FG3M",
    "fg3a": "FG3A",
    "ftm": "FTM",
    "fta": "FTA",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return float(value)
    except Exception:
        return default


def _season_start_year(season_id: Any) -> int:
    if isinstance(season_id, str):
        return int(season_id[:4])
    return int(season_id)


def _derive_advanced_from_totals(totals: Dict[str, float]) -> Dict[str, float]:
    fga = max(totals["fga"], 0.0)
    fgm = max(totals["fgm"], 0.0)
    fg3m = max(totals["fg3m"], 0.0)
    fta = max(totals["fta"], 0.0)
    ftm = max(totals["ftm"], 0.0)
    tov = max(totals["tov"], 0.0)

    efg_pct = (fgm + 0.5 * fg3m) / fga if fga > 0 else 0.0
    ts_den = 2.0 * (fga + 0.44 * fta)
    ts_pct = totals["pts"] / ts_den if ts_den > 0 else 0.0
    ast_tov = totals["ast"] / tov if tov > 0 else 0.0
    eff = (
        totals["pts"] + totals["reb"] + totals["ast"] + totals["stl"] + totals["blk"]
        - (fga - fgm)
        - (fta - ftm)
        - tov
    )

    return {
        "efg_pct": round(efg_pct, 3),
        "ts_pct": round(ts_pct, 3),
        "ast_to_tov": round(ast_tov, 2),
        "efficiency": round(eff, 1),
    }


def find_players_by_name(name: str, limit: int = 10) -> List[Dict[str, Any]]:
    matches = players.find_players_by_full_name(name)
    if not matches:
        return []

    out: List[Dict[str, Any]] = []
    for player in matches[:limit]:
        out.append(
            {
                "id": int(player["id"]),
                "full_name": player["full_name"],
                "is_active": bool(player.get("is_active", False)),
            }
        )
    return out


def find_player_id_by_name(name: str) -> Optional[int]:
    matches = find_players_by_name(name, limit=1)
    return matches[0]["id"] if matches else None


def get_name(player_id: int) -> Optional[str]:
    record = players.find_player_by_id(player_id)
    return record["full_name"] if record else None


def get_player_bio(player_id: int) -> Dict[str, Optional[Any]]:
    info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_dict()
    sets = info.get("resultSets", [])
    if not sets:
        return {"name": get_name(player_id), "position": None, "birth_date": None}
    headers = sets[0].get("headers", [])
    rows = sets[0].get("rowSet", [])
    if not rows:
        return {"name": get_name(player_id), "position": None, "birth_date": None}

    row = rows[0]
    mapping = {headers[i]: row[i] for i in range(min(len(headers), len(row)))}
    birth_date = mapping.get("BIRTHDATE")
    position = mapping.get("POSITION")
    display_name = mapping.get("DISPLAY_FIRST_LAST") or mapping.get("DISPLAY_LAST_COMMA_FIRST") or get_name(player_id)
    return {"name": display_name, "position": position, "birth_date": birth_date}


def get_player_age(player_id: int, season_year: Optional[int] = None) -> Optional[int]:
    bio = get_player_bio(player_id)
    birth_raw = bio.get("birth_date")
    return get_age_from_birth_date(birth_raw, season_year)


def get_age_from_birth_date(birth_raw: Any, season_year: Optional[int] = None) -> Optional[int]:
    if not birth_raw:
        return None

    try:
        birth = datetime.fromisoformat(str(birth_raw))
        if birth.tzinfo is None:
            birth = birth.replace(tzinfo=timezone.utc)
    except Exception:
        return None

    today = datetime(season_year, 6, 30, tzinfo=timezone.utc) if season_year else datetime.now(tz=timezone.utc)
    return ((today - birth) // 365).days


def get_player_position(player_id: int) -> Optional[str]:
    return get_player_bio(player_id).get("position")


def get_career_stats(player_id: int) -> List[Dict[str, Any]]:
    frame: pd.DataFrame = playercareerstats.PlayerCareerStats(player_id=player_id).get_data_frames()[0]
    if frame.empty:
        return []

    bio = get_player_bio(player_id)
    birth_date = bio.get("birth_date")
    data: List[Dict[str, Any]] = []
    for _, row in frame.iterrows():
        season_start = _season_start_year(row["SEASON_ID"])
        gp = max(_safe_float(row["GP"]), 1.0)
        totals = {key: _safe_float(row[col]) for key, col in TOTAL_COLUMNS.items()}

        per_game: Dict[str, float] = {}
        for key, col in PER_GAME_COLUMNS.items():
            if key.endswith("_pct"):
                per_game[key] = round(_safe_float(row[col]), 3)
            else:
                per_game[key] = round(_safe_float(row[col]) / gp, 1)

        derived = _derive_advanced_from_totals(totals)
        age = get_age_from_birth_date(birth_date, season_start)

        data.append(
            {
                "season_start": season_start,
                "season_label": row["SEASON_ID"],
                "age": age,
                "team_id": int(_safe_float(row.get("TEAM_ID", 0))),
                "gp": int(round(_safe_float(row["GP"]))),
                "gs": int(round(_safe_float(row.get("GS", 0)))),
                "per_game": per_game,
                "totals": {k: round(v, 1) for k, v in totals.items()},
                "advanced": derived,
            }
        )
    return data


def get_last_season_averages(player_id: int) -> Optional[Dict[str, Any]]:
    career = get_career_stats(player_id)
    if not career:
        return None
    latest = career[-1]
    return {
        "season": latest["season_start"],
        **latest["per_game"],
    }


def get_league_season_per_game(season_label: str, min_gp: int = 20) -> pd.DataFrame:
    frame = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season_label,
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]
    if frame.empty:
        return frame
    return frame[frame["GP"] >= min_gp].copy()



