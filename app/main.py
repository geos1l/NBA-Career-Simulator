from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from app.database import init_db
from app.schemas import CareerResponse, PlayerSearchResult, PlayerSummaryResponse, SimulateRequest
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
from app.services.sim import simulate_player_events

store = ModelStore()
_loaded_model = None

_NO_MODEL_MSG = (
    "No trained model found. Place a model under artifacts/models/simulator/ "
    "(see README) or train one with scripts/train.py."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loaded_model
    init_db()
    try:
        _loaded_model = store.load()
    except FileNotFoundError:
        pass  # model not yet trained — app starts but simulate will 503
    yield


app = FastAPI(title="NBA Career Simulator API", version="3.0.0", lifespan=lifespan)


@app.exception_handler(Exception)
def unhandled_exception_handler(request, exc):
    """Ensure every error response is JSON so the frontend never gets "is not valid JSON"."""
    from fastapi.responses import JSONResponse
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc) or "Internal server error"},
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model_only():
    """Return the model loaded at startup."""
    if _loaded_model is None:
        raise HTTPException(status_code=503, detail=_NO_MODEL_MSG)
    return _loaded_model


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/models/current")
def get_current_model() -> dict:
    loaded = load_model_only()
    return {"version": loaded.version, "metadata": loaded.metadata}


@app.get("/api/players/search", response_model=list[PlayerSearchResult])
def search_players(query: str) -> list[PlayerSearchResult]:
    if not query.strip():
        return []
    try:
        raw = find_players_by_name(query, limit=10)
        return [PlayerSearchResult(**p) for p in raw]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Player search failed: {e!s}") from e


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
    try:
        career = get_career_stats(player_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not load career: {e!s}") from e
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
    loaded = load_model_only()
    try:
        for msg in simulate_player_events(
            loaded_model=loaded,
            player_id=payload.player_id,
            start_season=payload.start_season,
            simulations=payload.simulations,
        ):
            if msg.get("event") == "complete":
                return msg["result"]
        raise RuntimeError("Simulation finished without a result")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/simulate/stream")
def simulate_stream(payload: SimulateRequest) -> StreamingResponse:
    """Server-Sent Events: real progress by completed Monte Carlo paths + ETA estimate."""

    def event_gen():
        try:
            loaded = load_model_only()
        except HTTPException as he:
            err = {"event": "error", "detail": he.detail, "status": he.status_code}
            yield f"data: {json.dumps(err)}\n\n"
            return

        try:
            for msg in simulate_player_events(
                loaded_model=loaded,
                player_id=payload.player_id,
                start_season=payload.start_season,
                simulations=payload.simulations,
            ):
                yield f"data: {json.dumps(msg)}\n\n"
        except ValueError as exc:
            yield f"data: {json.dumps({'event': 'error', 'detail': str(exc), 'status': 400})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'event': 'error', 'detail': f'Simulation failed: {exc!s}', 'status': 500})}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


frontend_dist = Path("frontend/dist")
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

    @app.get("/nba-logo.png")
    def nba_logo() -> FileResponse:
        return FileResponse(frontend_dist / "nba-logo.png")

    @app.get("/favicon.png")
    def favicon() -> FileResponse:
        return FileResponse(frontend_dist / "favicon.png")

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
