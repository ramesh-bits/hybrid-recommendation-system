"""
Content-Based Filtering using genre vectors.

User profile = weighted average of rated-item genre vectors (weight = rating).
Item scores = cosine similarity between user profile and item genre vector.
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Optional, Set


class ContentBasedFilter:
    def __init__(self, genre_cols: List[str]):
        self.genre_cols = genre_cols
        self.item_matrix: Optional[np.ndarray] = None   # (n_items, n_genres) float32
        self.item_idx_order: Optional[List[int]] = None  # item_idx at each row
        self.item_idx_to_row: dict = {}
        self.user_profiles: dict = {}  # user_idx → genre vector

    # ------------------------------------------------------------------
    def fit(self, ratings: pd.DataFrame, items: pd.DataFrame) -> None:
        # Build item feature matrix ordered by item_idx
        items_sorted = items.sort_values("item_idx").reset_index(drop=True)
        self.item_idx_order = items_sorted["item_idx"].tolist()
        self.item_idx_to_row = {idx: i for i, idx in enumerate(self.item_idx_order)}

        genre_matrix = items_sorted[self.genre_cols].values.astype(np.float32)
        # L2-normalise rows so cosine similarity = dot product
        norms = np.linalg.norm(genre_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.item_matrix = genre_matrix / norms

        # Build user profiles from training ratings
        self._build_user_profiles(ratings, genre_matrix)

    def _build_user_profiles(self, ratings: pd.DataFrame, genre_matrix: np.ndarray) -> None:
        for user_idx, group in ratings.groupby("user_idx"):
            self.user_profiles[user_idx] = self._compute_profile(group, genre_matrix)

    def _compute_profile(self, user_ratings: pd.DataFrame, genre_matrix: np.ndarray) -> np.ndarray:
        weights = user_ratings["rating"].values.astype(np.float32)
        rows = user_ratings["item_idx"].map(self.item_idx_to_row).values
        valid = ~pd.isna(user_ratings["item_idx"].map(self.item_idx_to_row))
        if not valid.any():
            return np.zeros(len(self.genre_cols), dtype=np.float32)
        vectors = genre_matrix[rows[valid.values]]
        weights_valid = weights[valid.values]
        profile = np.average(vectors, axis=0, weights=weights_valid)
        norm = np.linalg.norm(profile)
        return profile / norm if norm > 0 else profile

    # ------------------------------------------------------------------
    def predict_score(self, user_idx: int, item_idx: int) -> float:
        profile = self.user_profiles.get(user_idx)
        row = self.item_idx_to_row.get(item_idx)
        if profile is None or row is None:
            return 0.0
        return float(np.dot(profile, self.item_matrix[row]))

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        seen_items: Optional[Set[int]] = None,
    ) -> List[Tuple[int, float]]:
        profile = self.user_profiles.get(user_idx)
        if profile is None or self.item_matrix is None:
            return []

        scores = self.item_matrix @ profile  # (n_items,)

        if seen_items:
            for it in seen_items:
                r = self.item_idx_to_row.get(it)
                if r is not None:
                    scores[r] = -np.inf

        top_rows = np.argpartition(scores, -n)[-n:]
        top_rows = top_rows[np.argsort(scores[top_rows])[::-1]]
        return [(self.item_idx_order[r], float(scores[r])) for r in top_rows]

    # ------------------------------------------------------------------
    def update_profile(self, user_idx: int, item_idx: int, rating: float) -> None:
        """Real-time profile update when a user rates a new item."""
        row = self.item_idx_to_row.get(item_idx)
        if row is None:
            return
        item_vec = self.item_matrix[row]
        current = self.user_profiles.get(user_idx, np.zeros(len(self.genre_cols), dtype=np.float32))
        # Exponential moving average (α = 0.2) to incorporate new rating
        alpha = 0.2 * (rating / 5.0)
        updated = (1 - alpha) * current + alpha * item_vec
        norm = np.linalg.norm(updated)
        self.user_profiles[user_idx] = updated / norm if norm > 0 else updated

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "ContentBasedFilter":
        with open(path, "rb") as f:
            return pickle.load(f)
