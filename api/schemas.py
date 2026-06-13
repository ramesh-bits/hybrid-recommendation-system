from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    rank: int
    item_idx: int
    item_id: int
    title: str
    genres: List[str]
    score: float


class RecommendationResponse(BaseModel):
    user_idx: int
    model: str
    context: Dict[str, Any]
    recommendations: List[RecommendationItem]


class ExplanationResponse(BaseModel):
    user_idx: int
    item_idx: int
    item_id: int
    title: str
    model_contributions: Dict[str, Any]
    genre_match: Dict[str, float]
    similar_liked: List[Dict[str, Any]]
    natural_language: str


class SimulateRequest(BaseModel):
    user_idx: int
    item_idx: int
    rating: float = Field(ge=1.0, le=5.0)


class SimulateResponse(BaseModel):
    message: str
    user_idx: int
    item_idx: int
    updated_recommendations: List[RecommendationItem]


class EvaluationResponse(BaseModel):
    results: Dict[str, Dict[str, float]]
