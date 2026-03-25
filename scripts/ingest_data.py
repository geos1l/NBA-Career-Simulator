"""Ingest NBA player data from nba_api into the local SQLite database.

Usage:
    python scripts/ingest_data.py                  # All players (~5000, takes hours)
    python scripts/ingest_data.py --active-only     # Active players only (~600, ~10 min)
    python scripts/ingest_data.py --player-id 2544  # Single player (debug)
    python scripts/ingest_data.py --force           # Re-fetch even if data exists
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tqdm import tqdm

from nba_api.stats.endpoints import commonplayerinfo, playercareerstats
from nba_api.stats.static import players

from app.database import (
    _get_connection,
    init_db,
    set_meta,
    upsert_player,
    upsert_season_stat,
)
from app.services.external import (
    PER_GAME_COLUMNS,
    TOTAL_COLUMNS,
    _derive_advanced_from_totals,
    _safe_float,
    _season_start_year,
    get_age_from_birth_date,
)

NBA_STATS_TIMEOUT = 15
RATE_LIMIT_DELAY = 0.6  # seconds between API calls


def fetch_bio(player_id: int) -> Dict[str, Optional[str]]:
    """Fetch position and birth_date from CommonPlayerInfo API."""
    for attempt in range(2):
        try:
            info = commonplayerinfo.CommonPlayerInfo(
                player_id=player_id, timeout=NBA_STATS_TIMEOUT,
            ).get_dict()
            break
        except Exception as e:
            if "timed out" in str(e).lower() and attempt == 0:
                time.sleep(2)
                continue
            raise
    else:
        return {"position": None, "birth_date": None}

    sets = info.get("resultSets", [])
    if not sets:
        return {"position": None, "birth_date": None}
    headers = sets[0].get("headers", [])
    rows = sets[0].get("rowSet", [])
    if not rows:
        return {"position": None, "birth_date": None}

    row = rows[0]
    mapping = {headers[i]: row[i] for i in range(min(len(headers), len(row)))}
    return {
        "position": mapping.get("POSITION"),
        "birth_date": mapping.get("BIRTHDATE"),
    }


def fetch_career_rows(player_id: int) -> Optional[pd.DataFrame]:
    """Fetch career stats DataFrame from PlayerCareerStats API."""
    for attempt in range(2):
        try:
            frame = playercareerstats.PlayerCareerStats(
                player_id=player_id, timeout=NBA_STATS_TIMEOUT,
            ).get_data_frames()[0]
            return frame if not frame.empty else None
        except Exception as e:
            if "timed out" in str(e).lower() and attempt == 0:
                time.sleep(2)
                continue
            raise
    return None


def parse_season_row(
    player_id: int, row: pd.Series, birth_date: Optional[str]
) -> Dict[str, Any]:
    """Parse a single season row into the flat dict expected by upsert_season_stat."""
    season_start = _season_start_year(row["SEASON_ID"])
    gp = max(_safe_float(row["GP"]), 1.0)

    totals = {key: _safe_float(row[col]) for key, col in TOTAL_COLUMNS.items()}

    per_game: Dict[str, float] = {}
    for key, col in PER_GAME_COLUMNS.items():
        if key.endswith("_pct"):
            per_game[key] = round(_safe_float(row[col]), 3)
        else:
            per_game[key] = round(_safe_float(row[col]) / gp, 1)

    advanced = _derive_advanced_from_totals(totals)
    age = get_age_from_birth_date(birth_date, season_start)

    return {
        "player_id": player_id,
        "season_start": season_start,
        "season_label": row["SEASON_ID"],
        "team_id": int(_safe_float(row.get("TEAM_ID", 0))),
        "age": age,
        "gp": int(round(_safe_float(row["GP"]))),
        "gs": int(round(_safe_float(row.get("GS", 0)))),
        # Totals
        "pts_total": round(totals["pts"], 1),
        "reb_total": round(totals["reb"], 1),
        "ast_total": round(totals["ast"], 1),
        "stl_total": round(totals["stl"], 1),
        "blk_total": round(totals["blk"], 1),
        "tov_total": round(totals["tov"], 1),
        "min_total": round(totals["min"], 1),
        "fgm_total": round(totals["fgm"], 1),
        "fga_total": round(totals["fga"], 1),
        "fg3m_total": round(totals["fg3m"], 1),
        "fg3a_total": round(totals["fg3a"], 1),
        "ftm_total": round(totals["ftm"], 1),
        "fta_total": round(totals["fta"], 1),
        # Per-game
        **per_game,
        # Advanced
        **advanced,
    }


def ingest_player(player_id: int, name: str, first: str, last: str,
                   is_active: bool, force: bool, conn) -> bool:
    """Ingest a single player's bio + career stats. Returns True on success."""
    # Check if player already has stats and skip unless forced
    if not force:
        existing = conn.execute(
            "SELECT COUNT(*) as cnt FROM season_stats WHERE player_id = ?",
            (player_id,),
        ).fetchone()
        if existing and existing["cnt"] > 0:
            return True

    # Fetch bio
    try:
        bio = fetch_bio(player_id)
        time.sleep(RATE_LIMIT_DELAY)
    except Exception:
        bio = {"position": None, "birth_date": None}

    # Upsert player with bio data
    upsert_player(
        player_id=player_id,
        full_name=name,
        first_name=first,
        last_name=last,
        is_active=is_active,
        position=bio.get("position"),
        birth_date=bio.get("birth_date"),
        conn=conn,
    )

    # Fetch career stats
    try:
        frame = fetch_career_rows(player_id)
        time.sleep(RATE_LIMIT_DELAY)
    except Exception:
        conn.commit()
        return False

    if frame is None:
        conn.commit()
        return True  # Player exists but has no stats (very old / never played)

    # Parse and insert each season
    birth_date = bio.get("birth_date")
    for _, row in frame.iterrows():
        season_row = parse_season_row(player_id, row, birth_date)
        upsert_season_stat(season_row, conn=conn)

    conn.commit()
    return True


def main():
    parser = argparse.ArgumentParser(description="Ingest NBA player data into SQLite")
    parser.add_argument("--active-only", action="store_true",
                        help="Only fetch active players (~600)")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if data already exists")
    parser.add_argument("--player-id", type=int, default=None,
                        help="Fetch a single player by ID (debug)")
    args = parser.parse_args()

    init_db()
    conn = _get_connection()

    all_players = players.get_players()
    print(f"Total players in nba_api static list: {len(all_players)}")

    if args.player_id:
        # Single player mode
        target = [p for p in all_players if p["id"] == args.player_id]
        if not target:
            print(f"Player ID {args.player_id} not found in static list")
            conn.close()
            return
        player_list = target
    elif args.active_only:
        player_list = [p for p in all_players if p.get("is_active")]
        print(f"Active players: {len(player_list)}")
    else:
        player_list = all_players

    # First pass: insert all players into the players table (instant, no API calls)
    for p in all_players:
        upsert_player(
            player_id=p["id"],
            full_name=p["full_name"],
            first_name=p["first_name"],
            last_name=p["last_name"],
            is_active=bool(p.get("is_active", False)),
            conn=conn,
        )
    conn.commit()
    print(f"Inserted/updated {len(all_players)} players in players table")

    # Second pass: fetch career stats + bio for target players (API calls, slow)
    succeeded = 0
    failed = 0
    failed_ids = []

    for p in tqdm(player_list, desc="Fetching career stats"):
        try:
            ok = ingest_player(
                player_id=p["id"],
                name=p["full_name"],
                first=p["first_name"],
                last=p["last_name"],
                is_active=bool(p.get("is_active", False)),
                force=args.force,
                conn=conn,
            )
            if ok:
                succeeded += 1
            else:
                failed += 1
                failed_ids.append(p["id"])
        except Exception as e:
            failed += 1
            failed_ids.append(p["id"])
            tqdm.write(f"  Error for {p['full_name']} ({p['id']}): {e}")

    conn.close()

    # Save failed IDs for retry
    failed_path = Path(__file__).resolve().parent.parent / "data" / "failed_ingestions.json"
    if failed_ids:
        failed_path.write_text(json.dumps(failed_ids, indent=2))
        print(f"\nFailed player IDs saved to {failed_path}")

    set_meta("last_ingestion", datetime.now(tz=timezone.utc).isoformat())

    print(f"\nDone! Succeeded: {succeeded}, Failed: {failed}")
    print(f"Database: {Path(__file__).resolve().parent.parent / 'data' / 'nba.db'}")


if __name__ == "__main__":
    main()
