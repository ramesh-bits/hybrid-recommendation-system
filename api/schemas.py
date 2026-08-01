from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    rank:     int
    item_idx: int
    item_id:  Union[int, str]     # ints for MovieLens, ASIN strings for Amazon
    name:     str
    features: List[str]
    score:    float


class RecommendationResponse(BaseModel):
    user_idx:        int
    model:           str
    context:         Dict[str, Any]
    recommendations: List[RecommendationItem]


class ExplanationResponse(BaseModel):
    user_idx:            int
    item_idx:            int
    item_id:             Union[int, str]
    name:                str
    model_contributions: Dict[str, Any]
    feature_match:       Dict[str, float]
    similar_liked:       List[Dict[str, Any]]
    natural_language:    str


class SimulateRequest(BaseModel):
    user_idx: int
    item_idx: int
    rating:   Optional[float] = Field(None, ge=1.0, le=5.0,
                                      description="Explicit rating; optional if behavioural events are set")
    # Behavioural signals (simulated from the dashboard)
    view_time_sec: float = Field(0.0, ge=0.0, description="Seconds spent viewing the item")
    clicked:       bool  = False
    liked:         bool  = False
    wishlist:      bool  = False
    add_to_cart:   bool  = False
    ordered:       bool  = False


class SimulateResponse(BaseModel):
    message:                 str
    user_idx:                int
    item_idx:                int
    engagement_weight:       float
    effective_rating:        float
    profile_update_alpha:    float
    signals_applied:         List[str]
    updated_recommendations: List[RecommendationItem]


class EvaluationResponse(BaseModel):
    results: Dict[str, Dict[str, float]]


class DatasetInfoResponse(BaseModel):
    dataset_name:  str
    item_label:    str
    feature_cols:  List[str]
    feature_label: str
    n_users:       int
    n_items:       int
