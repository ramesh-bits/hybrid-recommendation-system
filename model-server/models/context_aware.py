"""
Context-Aware module.

Learns per-genre temporal preferences (hour-of-day, day-of-week) from
training data. At inference time it returns a context adjustment score
that re-weights candidate items based on when the recommendation is made.
"""
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


class ContextAwareAdjuster:
    def __init__(self, genre_cols: List[str]):
        self.genre_cols = genre_cols
        # genre → {hour: mean_rating}, {dow: mean_rating}
        self.hour_bias: dict = {}
        self.dow_bias: dict = {}
        self.global_hour_bias: dict = {}
        self.global_dow_bias: dict = {}

    # ------------------------------------------------------------------
    def fit(self, ratings: pd.DataFrame, items: pd.DataFrame) -> None:
        merged = ratings.merge(
            items[["item_idx"] + self.genre_cols], on="item_idx", how="left"
        )

        # Global temporal biases (rating ~ hour, dow)
        self.global_hour_bias = (
            merged.groupby("hour")["rating"].mean() - merged["rating"].mean()
        ).to_dict()
        self.global_dow_bias = (
            merged.groupby("day_of_week")["rating"].mean() - merged["rating"].mean()
        ).to_dict()

        # Per-genre biases
        global_mean = merged["rating"].mean()
        for genre in self.genre_cols:
            g_df = merged[merged[genre] == 1]
            if len(g_df) < 10:
                continue
            self.hour_bias[genre] = (
                g_df.groupby("hour")["rating"].mean() - global_mean
            ).to_dict()
            self.dow_bias[genre] = (
                g_df.groupby("day_of_week")["rating"].mean() - global_mean
            ).to_dict()

    # ------------------------------------------------------------------
    def score(self, item_genres: dict, hour: int, dow: int) -> float:
        """Return context adjustment in [-1, 1] for (item, time) pair."""
        genre_scores = []
        for genre, flag in item_genres.items():
            if flag != 1 or genre not in self.hour_bias:
                continue
            h = self.hour_bias[genre].get(hour, 0.0)
            d = self.dow_bias[genre].get(dow, 0.0)
            genre_scores.append((h + d) / 2.0)

        if not genre_scores:
            h = self.global_hour_bias.get(hour, 0.0)
            d = self.global_dow_bias.get(dow, 0.0)
            return (h + d) / 2.0

        return float(np.mean(genre_scores))

    def adjust_scores(
        self,
        scores: List[Tuple[int, float]],
        items: pd.DataFrame,
        item_idx_col: str = "item_idx",
        hour: int = 12,
        dow: int = 2,
        alpha: float = 0.15,
    ) -> List[Tuple[int, float]]:
        """Apply context adjustment to a ranked list of (item_idx, score) pairs."""
        genre_lookup = (
            items.set_index(item_idx_col)[self.genre_cols].to_dict(orient="index")
        )
        adjusted = []
        for item_idx, base_score in scores:
            genres = genre_lookup.get(item_idx, {})
            ctx = self.score(genres, hour, dow)
            adjusted.append((item_idx, base_score + alpha * ctx))
        return sorted(adjusted, key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "ContextAwareAdjuster":
        with open(path, "rb") as f:
            return pickle.load(f)
