from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from app.schemas import CareerResponse, PlayerSearchResult, PlayerSummaryResponse, RetrainRequest, SimulateRequest
from app.services.external import (
    find_player_id_by_name,
    find_players_by_name,
    get_career_stats,
    get_last_season_averages,
    get_name,
    get_player_age,
    get_player_position,
)
from app.services.model_store import ModelStore
from app.services.sim import simulate_player
from app.services.trainer import train_model

app = FastAPI(title="NBA Player Simulator API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store = ModelStore()


def load_or_train_model():
    try:
        return store.load()
    except FileNotFoundError:
        output = train_model()
        metadata = store.save(output.artifact, output.metadata)
        return store.load(metadata["version"])


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/models/current")
def get_current_model() -> dict:
    loaded = load_or_train_model()
    return {"version": loaded.version, "metadata": loaded.metadata}


def _json_safe(obj: dict) -> dict:
    """Return a copy of the dict with only JSON-serializable values (no numpy, Path, etc.)."""

    def _val(x):
        if isinstance(x, dict):
            return _json_safe(x)
        if isinstance(x, (list, tuple)):
            return [_val(i) for i in x]
        if hasattr(x, "item"):  # numpy scalar
            return x.item()
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        return str(x)

    return {k: _val(v) for k, v in obj.items()}


@app.post("/api/models/retrain")
def retrain_model(payload: RetrainRequest) -> dict:
    try:
        output = train_model(start_year=payload.start_year, end_year=payload.end_year)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e!s}") from e
    try:
        metadata = store.save(output.artifact, output.metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save failed: {e!s}") from e
    try:
        safe_meta = _json_safe(metadata)
        return {"status": "retrained", "version": metadata["version"], "metadata": safe_meta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model saved but response failed: {e!s}") from e


@app.get("/api/players/search", response_model=list[PlayerSearchResult])
def search_players(query: str) -> list[PlayerSearchResult]:
    if not query.strip():
        return []
    return [PlayerSearchResult(**p) for p in find_players_by_name(query, limit=10)]


@app.get("/api/player", response_model=PlayerSummaryResponse)
def get_player_summary(name: str) -> PlayerSummaryResponse:
    player_id = find_player_id_by_name(name)
    if player_id is None:
        raise HTTPException(status_code=404, detail="Player not found")

    stats = get_last_season_averages(player_id)
    if stats is None:
        raise HTTPException(status_code=404, detail="Stats not found")

    age = get_player_age(player_id, stats["season"])
    position = get_player_position(player_id)
    player_name = get_name(player_id)
    return PlayerSummaryResponse(
        player_id=player_id,
        name=player_name,
        age=age,
        position=position,
        latest_season=stats["season"],
        latest_per_game=stats,
    )


@app.get("/api/players/{player_id}/career", response_model=CareerResponse)
def get_player_career(player_id: int) -> CareerResponse:
    career = get_career_stats(player_id)
    if not career:
        raise HTTPException(status_code=404, detail="Career stats not found")
    return CareerResponse(
        player_id=player_id,
        name=get_name(player_id),
        position=get_player_position(player_id),
        seasons_played=len(career),
        seasons=career,
    )


@app.post("/api/simulate")
def simulate(payload: SimulateRequest) -> dict:
    loaded = load_or_train_model()
    try:
        result = simulate_player(
            loaded_model=loaded,
            player_id=payload.player_id,
            start_season=payload.start_season,
            simulations=payload.simulations,
            realism_profile=payload.realism_profile,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


frontend_dist = Path("frontend/dist")
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

    @app.get("/")
    def home() -> FileResponse:
        return FileResponse(frontend_dist / "index.html")

else:
    @app.get("/")
    def home_dev() -> JSONResponse:
        return JSONResponse(
            {
                "message": "Frontend build not found. Run `npm install && npm run build` inside `frontend/`.",
            }
        )

