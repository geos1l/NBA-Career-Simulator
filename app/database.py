"""SQLite database layer for NBA player data.

All query functions return data in the exact same shapes as the original
external.py functions so that sim.py and main.py need zero changes.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "nba.db"


def _get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = _get_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS players (
                player_id   INTEGER PRIMARY KEY,
                full_name   TEXT NOT NULL,
                first_name  TEXT,
                last_name   TEXT,
                is_active   INTEGER NOT NULL DEFAULT 0,
                position    TEXT,
                birth_date  TEXT,
                updated_at  TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_players_name
                ON players(full_name COLLATE NOCASE);

            CREATE TABLE IF NOT EXISTS season_stats (
                player_id    INTEGER NOT NULL,
                season_start INTEGER NOT NULL,
                season_label TEXT NOT NULL,
                team_id      INTEGER,
                age          INTEGER,
                gp           INTEGER,
                gs           INTEGER,
                -- Totals
                pts_total    REAL, reb_total    REAL, ast_total    REAL,
                stl_total    REAL, blk_total    REAL, tov_total    REAL,
                min_total    REAL, fgm_total    REAL, fga_total    REAL,
                fg3m_total   REAL, fg3a_total   REAL, ftm_total    REAL,
                fta_total    REAL,
                -- Per-game
                ppg REAL, rpg REAL, apg REAL, spg REAL, bpg REAL,
                tpg REAL, mpg REAL,
                fg_pct REAL, fg3_pct REAL, ft_pct REAL,
                -- Advanced
                efg_pct REAL, ts_pct REAL, ast_to_tov REAL, efficiency REAL,

                PRIMARY KEY (player_id, season_start),
                FOREIGN KEY (player_id) REFERENCES players(player_id)
            );

            CREATE TABLE IF NOT EXISTS ingestion_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Query functions — return shapes match external.py exactly
# ---------------------------------------------------------------------------

def search_players(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Return list of {id, full_name, is_active} matching query."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT player_id, full_name, is_active FROM players "
            "WHERE full_name LIKE ? COLLATE NOCASE ORDER BY is_active DESC, full_name LIMIT ?",
            (f"%{query}%", limit),
        ).fetchall()
        return [
            {"id": r["player_id"], "full_name": r["full_name"], "is_active": bool(r["is_active"])}
            for r in rows
        ]
    finally:
        conn.close()


def find_player_id_by_name(name: str) -> Optional[int]:
    results = search_players(name, limit=1)
    return results[0]["id"] if results else None


def get_name(player_id: int) -> Optional[str]:
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT full_name FROM players WHERE player_id = ?", (player_id,)
        ).fetchone()
        return row["full_name"] if row else None
    finally:
        conn.close()


def get_player_bio(player_id: int) -> Dict[str, Optional[Any]]:
    """Return {name, position, birth_date} — same shape as external.get_player_bio."""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT full_name, position, birth_date FROM players WHERE player_id = ?",
            (player_id,),
        ).fetchone()
        if not row:
            return {"name": None, "position": None, "birth_date": None}
        return {
            "name": row["full_name"],
            "position": row["position"],
            "birth_date": row["birth_date"],
        }
    finally:
        conn.close()


def get_player_age(player_id: int, season_year: Optional[int] = None) -> Optional[int]:
    bio = get_player_bio(player_id)
    birth_raw = bio.get("birth_date")
    return _age_from_birth_date(birth_raw, season_year)


def get_player_position(player_id: int) -> Optional[str]:
    return get_player_bio(player_id).get("position")


def _age_from_birth_date(birth_raw: Any, season_year: Optional[int] = None) -> Optional[int]:
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


def get_career_stats(player_id: int) -> List[Dict[str, Any]]:
    """Return list of season dicts with nested per_game/totals/advanced — same shape as external."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM season_stats WHERE player_id = ? ORDER BY season_start",
            (player_id,),
        ).fetchall()
        if not rows:
            return []

        data: List[Dict[str, Any]] = []
        for r in rows:
            per_game = {
                "ppg": r["ppg"], "rpg": r["rpg"], "apg": r["apg"],
                "spg": r["spg"], "bpg": r["bpg"], "tpg": r["tpg"],
                "mpg": r["mpg"], "fg_pct": r["fg_pct"],
                "fg3_pct": r["fg3_pct"], "ft_pct": r["ft_pct"],
            }
            totals = {
                "pts": r["pts_total"], "reb": r["reb_total"], "ast": r["ast_total"],
                "stl": r["stl_total"], "blk": r["blk_total"], "tov": r["tov_total"],
                "min": r["min_total"], "fgm": r["fgm_total"], "fga": r["fga_total"],
                "fg3m": r["fg3m_total"], "fg3a": r["fg3a_total"],
                "ftm": r["ftm_total"], "fta": r["fta_total"],
            }
            advanced = {
                "efg_pct": r["efg_pct"], "ts_pct": r["ts_pct"],
                "ast_to_tov": r["ast_to_tov"], "efficiency": r["efficiency"],
            }
            data.append({
                "season_start": r["season_start"],
                "season_label": r["season_label"],
                "age": r["age"],
                "team_id": r["team_id"],
                "gp": r["gp"],
                "gs": r["gs"],
                "per_game": per_game,
                "totals": totals,
                "advanced": advanced,
            })
        return data
    finally:
        conn.close()


def get_last_season_averages(player_id: int) -> Optional[Dict[str, Any]]:
    career = get_career_stats(player_id)
    if not career:
        return None
    latest = career[-1]
    return {
        "season": latest["season_start"],
        **latest["per_game"],
    }


# ---------------------------------------------------------------------------
# Write helpers — used by the ingestion script
# ---------------------------------------------------------------------------

def upsert_player(
    player_id: int,
    full_name: str,
    first_name: str,
    last_name: str,
    is_active: bool,
    position: Optional[str] = None,
    birth_date: Optional[str] = None,
    *,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    _conn = conn or _get_connection()
    try:
        _conn.execute(
            """INSERT INTO players (player_id, full_name, first_name, last_name, is_active, position, birth_date, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(player_id) DO UPDATE SET
                   full_name=excluded.full_name, first_name=excluded.first_name,
                   last_name=excluded.last_name, is_active=excluded.is_active,
                   position=COALESCE(excluded.position, players.position),
                   birth_date=COALESCE(excluded.birth_date, players.birth_date),
                   updated_at=excluded.updated_at""",
            (player_id, full_name, first_name, last_name, int(is_active),
             position, birth_date, datetime.now(tz=timezone.utc).isoformat()),
        )
        if conn is None:
            _conn.commit()
    finally:
        if conn is None:
            _conn.close()


def upsert_season_stat(row: Dict[str, Any], *, conn: Optional[sqlite3.Connection] = None) -> None:
    _conn = conn or _get_connection()
    try:
        _conn.execute(
            """INSERT INTO season_stats (
                   player_id, season_start, season_label, team_id, age, gp, gs,
                   pts_total, reb_total, ast_total, stl_total, blk_total, tov_total,
                   min_total, fgm_total, fga_total, fg3m_total, fg3a_total, ftm_total, fta_total,
                   ppg, rpg, apg, spg, bpg, tpg, mpg, fg_pct, fg3_pct, ft_pct,
                   efg_pct, ts_pct, ast_to_tov, efficiency
               ) VALUES (
                   :player_id, :season_start, :season_label, :team_id, :age, :gp, :gs,
                   :pts_total, :reb_total, :ast_total, :stl_total, :blk_total, :tov_total,
                   :min_total, :fgm_total, :fga_total, :fg3m_total, :fg3a_total, :ftm_total, :fta_total,
                   :ppg, :rpg, :apg, :spg, :bpg, :tpg, :mpg, :fg_pct, :fg3_pct, :ft_pct,
                   :efg_pct, :ts_pct, :ast_to_tov, :efficiency
               ) ON CONFLICT(player_id, season_start) DO UPDATE SET
                   season_label=excluded.season_label, team_id=excluded.team_id,
                   age=excluded.age, gp=excluded.gp, gs=excluded.gs,
                   pts_total=excluded.pts_total, reb_total=excluded.reb_total,
                   ast_total=excluded.ast_total, stl_total=excluded.stl_total,
                   blk_total=excluded.blk_total, tov_total=excluded.tov_total,
                   min_total=excluded.min_total, fgm_total=excluded.fgm_total,
                   fga_total=excluded.fga_total, fg3m_total=excluded.fg3m_total,
                   fg3a_total=excluded.fg3a_total, ftm_total=excluded.ftm_total,
                   fta_total=excluded.fta_total,
                   ppg=excluded.ppg, rpg=excluded.rpg, apg=excluded.apg,
                   spg=excluded.spg, bpg=excluded.bpg, tpg=excluded.tpg,
                   mpg=excluded.mpg, fg_pct=excluded.fg_pct, fg3_pct=excluded.fg3_pct,
                   ft_pct=excluded.ft_pct, efg_pct=excluded.efg_pct, ts_pct=excluded.ts_pct,
                   ast_to_tov=excluded.ast_to_tov, efficiency=excluded.efficiency""",
            row,
        )
        if conn is None:
            _conn.commit()
    finally:
        if conn is None:
            _conn.close()


def set_meta(key: str, value: str, *, conn: Optional[sqlite3.Connection] = None) -> None:
    _conn = conn or _get_connection()
    try:
        _conn.execute(
            "INSERT INTO ingestion_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        if conn is None:
            _conn.commit()
    finally:
        if conn is None:
            _conn.close()


def get_meta(key: str) -> Optional[str]:
    conn = _get_connection()
    try:
        row = conn.execute("SELECT value FROM ingestion_meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None
    finally:
        conn.close()
