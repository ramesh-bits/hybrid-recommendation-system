"""
Context-Aware module.

Learns per-feature temporal preferences (hour-of-day, day-of-week) from
training data. At inference time it returns a context adjustment score
that re-weights candidate items based on when the recommendation is made.
"""
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


class ContextAwareAdjuster:
    def __init__(self, feature_cols: List[str]):
        self.feature_cols = feature_cols
        self.hour_bias: dict = {}
        self.dow_bias:  dict = {}
        self.global_hour_bias: dict = {}
        self.global_dow_bias:  dict = {}

    def __setstate__(self, state):
        # Migrate pickles saved with old attribute name genre_cols
        if "genre_cols" in state and "feature_cols" not in state:
            state["feature_cols"] = state.pop("genre_cols")
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    def fit(self, ratings: pd.DataFrame, items: pd.DataFrame) -> None:
        merged = ratings.merge(
            items[["item_idx"] + self.feature_cols], on="item_idx", how="left"
        )

        global_mean = merged["rating"].mean()
        self.global_hour_bias = (
            merged.groupby("hour")["rating"].mean() - global_mean
        ).to_dict()
        self.global_dow_bias = (
            merged.groupby("day_of_week")["rating"].mean() - global_mean
        ).to_dict()

        for feat in self.feature_cols:
            f_df = merged[merged[feat] == 1]
            if len(f_df) < 10:
                continue
            self.hour_bias[feat] = (
                f_df.groupby("hour")["rating"].mean() - global_mean
            ).to_dict()
            self.dow_bias[feat] = (
                f_df.groupby("day_of_week")["rating"].mean() - global_mean
            ).to_dict()

    # ------------------------------------------------------------------
    def score(self, item_features: dict, hour: int, dow: int) -> float:
        """Return context adjustment in [-1, 1] for (item, time) pair."""
        feat_scores = []
        for feat, flag in item_features.items():
            if flag != 1 or feat not in self.hour_bias:
                continue
            h = self.hour_bias[feat].get(hour, 0.0)
            d = self.dow_bias[feat].get(dow, 0.0)
            feat_scores.append((h + d) / 2.0)

        if not feat_scores:
            h = self.global_hour_bias.get(hour, 0.0)
            d = self.global_dow_bias.get(dow, 0.0)
            return (h + d) / 2.0

        return float(np.mean(feat_scores))

    def adjust_scores(
        self,
        scores: List[Tuple[int, float]],
        items: pd.DataFrame,
        item_idx_col: str = "item_idx",
        hour: int = 12,
        dow: int = 2,
        alpha: float = 0.15,
    ) -> List[Tuple[int, float]]:
        feat_lookup = (
            items.set_index(item_idx_col)[self.feature_cols].to_dict(orient="index")
        )
        adjusted = []
        for item_idx, base_score in scores:
            feats = feat_lookup.get(item_idx, {})
            ctx   = self.score(feats, hour, dow)
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
