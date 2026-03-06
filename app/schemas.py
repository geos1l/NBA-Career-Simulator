from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


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
    start_year: int = Field(
        default=2000,
        ge=1979,
        le=2050,
        description="First season to include (e.g. 2000 = from 2000-01 onward). Must be <= end_year.",
    )
    end_year: Optional[int] = Field(
        default=None,
        ge=1979,
        le=2055,
        description="Last season to include. Omit for 'current year - 2'. Must be >= start_year.",
    )

    @model_validator(mode="after")
    def start_before_end(self) -> "RetrainRequest":
        if self.end_year is not None and self.start_year > self.end_year:
            raise ValueError("start_year must be <= end_year")
        return self

