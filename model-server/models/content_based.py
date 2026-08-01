"""
Content-Based Filtering using item feature vectors.

User profile = weighted average of LIKED-item feature vectors (weight = rating).
Item scores  = cosine similarity between user profile and item feature vector,
               plus a small log-popularity term that breaks ties between the
               many items sharing an identical feature (genre) signature.

Ranking quality fixes (v2):
  - Profiles are built only from items the user rated at or above
    like_threshold. Previously 1-star items contributed positively
    (weight = rating), pulling the profile towards disliked features.
  - Low-dimensional binary feature vectors produce large groups of items
    with identical cosine similarity; pop_weight * log1p(popularity)
    orders items sensibly within those ties.
  - Optional interaction_weight column scales each rating's contribution.
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Optional, Set


class ContentBasedFilter:
    def __init__(
        self,
        feature_cols: List[str],
        like_threshold: float = 3.5,
        pop_weight: float = 0.05,
    ):
        self.feature_cols = feature_cols
        self.like_threshold = like_threshold
        self.pop_weight = pop_weight
        self.item_matrix: Optional[np.ndarray] = None   # (n_items, n_features) float32
        self.item_idx_order: Optional[List[int]] = None
        self.item_idx_to_row: dict = {}
        self.user_profiles: dict = {}  # user_idx → feature vector
        self.pop_scores: Optional[np.ndarray] = None    # log1p popularity per row

    def __setstate__(self, state):
        # Migrate pickles saved with old attribute name genre_cols
        if "genre_cols" in state and "feature_cols" not in state:
            state["feature_cols"] = state.pop("genre_cols")
        state.setdefault("like_threshold", 3.5)
        state.setdefault("pop_weight", 0.05)
        state.setdefault("pop_scores", None)
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    def fit(self, ratings: pd.DataFrame, items: pd.DataFrame) -> None:
        items_sorted = items.sort_values("item_idx").reset_index(drop=True)
        self.item_idx_order = items_sorted["item_idx"].tolist()
        self.item_idx_to_row = {idx: i for i, idx in enumerate(self.item_idx_order)}

        feature_matrix = items_sorted[self.feature_cols].values.astype(np.float32)
        norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.item_matrix = feature_matrix / norms

        # Popularity per item row (weighted by interaction_weight if present)
        if "interaction_weight" in ratings.columns:
            pop = ratings.groupby("item_idx")["interaction_weight"].sum()
        else:
            pop = ratings.groupby("item_idx").size()
        self.pop_scores = np.array(
            [np.log1p(pop.get(it, 0.0)) for it in self.item_idx_order],
            dtype=np.float32,
        )

        self._build_user_profiles(ratings)

    def _build_user_profiles(self, ratings: pd.DataFrame) -> None:
        for user_idx, group in ratings.groupby("user_idx"):
            self.user_profiles[user_idx] = self._compute_profile(group)

    def _compute_profile(self, user_ratings: pd.DataFrame) -> np.ndarray:
        # Only liked items shape the profile; fall back to all if none liked
        liked = user_ratings[user_ratings["rating"] >= self.like_threshold]
        src = liked if len(liked) > 0 else user_ratings

        rows, weights = [], []
        has_iw = "interaction_weight" in src.columns
        for _, rec in src.iterrows():
            row = self.item_idx_to_row.get(rec["item_idx"])
            if row is None:
                continue
            rows.append(row)
            w = float(rec["rating"])
            if has_iw and not pd.isna(rec.get("interaction_weight")):
                w *= float(rec["interaction_weight"])
            weights.append(w)

        if not rows:
            return np.zeros(len(self.feature_cols), dtype=np.float32)
        profile = np.average(self.item_matrix[rows], axis=0, weights=weights)
        norm = np.linalg.norm(profile)
        return profile / norm if norm > 0 else profile

    # ------------------------------------------------------------------
    def predict_score(self, user_idx: int, item_idx: int) -> float:
        profile = self.user_profiles.get(user_idx)
        row     = self.item_idx_to_row.get(item_idx)
        if profile is None or row is None:
            return 0.0
        return float(np.dot(profile, self.item_matrix[row]))

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        seen_items: Optional[Set[int]] = None,
        profile_override: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, float]]:
        """profile_override lets the caller score against a temporary
        (e.g. session-context-adjusted) profile without persisting it."""
        profile = profile_override if profile_override is not None \
            else self.user_profiles.get(user_idx)
        if profile is None or self.item_matrix is None:
            return []

        scores = self.item_matrix @ profile
        if self.pop_scores is not None and self.pop_weight > 0:
            scores = scores + self.pop_weight * self.pop_scores

        if seen_items:
            for it in seen_items:
                r = self.item_idx_to_row.get(it)
                if r is not None:
                    scores[r] = -np.inf

        top_rows = np.argpartition(scores, -n)[-n:]
        top_rows = top_rows[np.argsort(scores[top_rows])[::-1]]
        return [(self.item_idx_order[r], float(scores[r])) for r in top_rows]

    # ------------------------------------------------------------------
    def _ema_profile(
        self, current: np.ndarray, item_idx: int, rating: float, engagement_weight: float
    ) -> Tuple[np.ndarray, float]:
        """Pure EMA step: returns (updated profile, alpha used). No state change."""
        row = self.item_idx_to_row.get(item_idx)
        if row is None:
            return current, 0.0
        item_vec = self.item_matrix[row]
        alpha    = min(0.6, 0.2 * (rating / 5.0) * max(engagement_weight, 0.1))
        updated  = (1 - alpha) * current + alpha * item_vec
        norm     = np.linalg.norm(updated)
        return (updated / norm if norm > 0 else updated), float(alpha)

    def preview_profile(
        self, user_idx: int, item_idx: int, rating: float, engagement_weight: float = 1.0
    ) -> Optional[np.ndarray]:
        """Session-context profile: the EMA-updated profile WITHOUT persisting.
        Used to answer 'what would this user see right after this interaction'."""
        current = self.user_profiles.get(
            user_idx, np.zeros(len(self.feature_cols), dtype=np.float32))
        updated, alpha = self._ema_profile(current, item_idx, rating, engagement_weight)
        return updated if alpha > 0 else None

    def update_profile(
        self,
        user_idx: int,
        item_idx: int,
        rating: float,
        engagement_weight: float = 1.0,
    ) -> float:
        """
        Real-time EMA profile update when a user rates or interacts with an item.

        engagement_weight (>= 1.0) comes from behavioural signals (view time,
        clicks, add-to-cart, wishlist, orders) and amplifies the step size:
        a purchase moves the profile more than a plain rating.
        Returns the effective alpha used, for display in the dashboard.
        """
        current = self.user_profiles.get(
            user_idx, np.zeros(len(self.feature_cols), dtype=np.float32))
        updated, alpha = self._ema_profile(current, item_idx, rating, engagement_weight)
        if alpha > 0:
            self.user_profiles[user_idx] = updated
        return alpha

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "ContentBasedFilter":
        with open(path, "rb") as f:
            return pickle.load(f)
