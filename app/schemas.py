from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PlayerSearchResult(BaseModel):
    id: int
    full_name: str
    is_active: bool


class PlayerSummaryResponse(BaseModel):
    player_id: int
    name: Optional[str]
    age: Optional[int]
    position: Optional[str]
    latest_season: Optional[int]
    latest_per_game: Dict[str, Any]


class CareerResponse(BaseModel):
    player_id: int
    name: Optional[str]
    position: Optional[str]
    seasons_played: int
    seasons: List[Dict[str, Any]]


class SimulateRequest(BaseModel):
    player_id: int
    start_season: int = Field(..., description="Starting season year (e.g., 2019)")
    simulations: int = Field(default=250, ge=50, le=1000)
    realism_profile: str = Field(default="realistic")


class RetrainRequest(BaseModel):
    start_year: int = Field(default=2000, ge=1979, le=2050)
    end_year: Optional[int] = Field(default=None, ge=1979, le=2055)

