"""
Pydantic models for PermitIQ API request/response schemas.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class ProposalInput(BaseModel):
    parcel_id: str = Field(default="", description="Boston parcel ID for zoning lookup")
    proposed_use: str = Field(default="", alias="use_type", description="Proposed use type (also accepts 'use_type')")
    variances: List[str] = Field(default_factory=list, description="List of variance types requested")
    project_type: Optional[str] = Field(default=None, description="Project type (addition, new_construction, etc.)")
    ward: Optional[str] = Field(default=None, description="Boston ward number")
    has_attorney: bool = Field(default=False, description="Whether project has legal representation")
    proposed_units: int = Field(default=0, ge=0, le=500, description="Number of proposed units")
    proposed_stories: int = Field(default=0, ge=0, le=50, description="Number of proposed stories")

    class Config:
        populate_by_name = True


class HealthResponse(BaseModel):
    status: str
    geojson_loaded: bool
    zba_loaded: bool
    model_loaded: bool
    total_parcels: int
    total_cases: int
    model_name: Optional[str]
    model_auc: Optional[float]
    features: int


class SearchResult(BaseModel):
    address: str
    ward: str = ""
    zoning: str = ""
    total_cases: int
    approved: int
    denied: int
    approval_rate: Optional[float]
    latest_date: str = ""
    latest_case: str = ""


class PredictionResponse(BaseModel):
    parcel_id: str
    zoning: str
    district: str
    proposed_use: str
    project_type: str
    variances: List[str]
    has_attorney: bool
    approval_probability: float
    probability_range: List[float]
    confidence: str
    based_on_cases: int
    ward_approval_rate: Optional[float]
    key_factors: List[str]
    top_drivers: list
    similar_cases: list
    estimated_timeline_days: Optional[dict]
    model: str
    model_auc: float = 0
    total_training_cases: int = 0
    disclaimer: str


class WardStatsResponse(BaseModel):
    ward: str
    total_cases: int
    approved: int
    denied: int
    approval_rate: float


class RecommendationResult(BaseModel):
    parcel_id: str
    approval_probability: float
    zoning_code: str = ""
    district: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None


class RecommendationResponse(BaseModel):
    query: dict
    total_candidates: int
    results_found: int
    parcels: List[RecommendationResult]
    disclaimer: str
