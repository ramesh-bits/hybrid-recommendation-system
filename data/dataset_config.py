"""
DatasetConfig — canonical schema for a dataset loaded into the recommender.

Stored inside meta.pkl so every service reads it from a single source.
Used as a Pydantic model for validation at ingest and upload time.
"""
from typing import List
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    dataset_name: str = Field(..., description="Unique slug, e.g. 'movielens-100k'")
    item_label: str = Field("item", description="Singular noun for an item, e.g. 'movie', 'product', 'song'")
    feature_cols: List[str] = Field(..., description="Binary 0/1 category columns present in items.csv")
    feature_label: str = Field("Category", description="Display name for the feature dimension, e.g. 'Genre', 'Category'")
    rating_threshold: float = Field(3.5, description="Min rating to treat as implicit positive label")
    item_name_col: str = Field("name", description="Column in items.csv holding the display name")

    def to_meta_dict(self) -> dict:
        """Serialise into the meta.pkl dict that the model-server reads."""
        return {
            "dataset_name":    self.dataset_name,
            "item_label":      self.item_label,
            "feature_cols":    self.feature_cols,
            "feature_label":   self.feature_label,
            "rating_threshold": self.rating_threshold,
            "item_name_col":   self.item_name_col,
            # backwards-compat alias — removed after next make data run
            "genre_cols":      self.feature_cols,
        }
