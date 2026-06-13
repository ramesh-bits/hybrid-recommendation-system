"""
Explainability engine for the hybrid recommender.

Produces four complementary explanation signals per (user, item) pair:

  1. model_contributions  – raw and normalised score from each sub-model
  2. genre_match          – how well the item's genres align with the user's
                            historical genre preferences
  3. similar_liked        – items the user rated highly that share genres
  4. natural_language     – one-sentence explanation combining the above

SHAP-based feature attribution for NCF is provided separately via
`NcfShapExplainer` (requires `shap` package).
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

from models.collaborative_filtering import SVDCollaborativeFilter
from models.content_based import ContentBasedFilter
from models.neural_cf import NeuralCollaborativeFilter


class HybridExplainer:
    def __init__(
        self,
        cf: SVDCollaborativeFilter,
        cbf: ContentBasedFilter,
        ncf: NeuralCollaborativeFilter,
        items_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        genre_cols: List[str],
        hybrid_weights=(0.30, 0.30, 0.40),
    ):
        self.cf = cf
        self.cbf = cbf
        self.ncf = ncf
        self.items_df = items_df
        self.ratings_df = ratings_df
        self.genre_cols = genre_cols
        self.weights = hybrid_weights

        # Pre-compute user liked-item sets for speed
        self._user_liked: Dict[int, List[int]] = (
            ratings_df[ratings_df["rating"] >= 4.0]
            .groupby("user_idx")["item_idx"]
            .apply(list)
            .to_dict()
        )
        self._item_id_to_idx = dict(zip(items_df["item_id"], items_df["item_idx"]))
        self._item_idx_to_row = {idx: i for i, idx in enumerate(items_df["item_idx"])}

    # ------------------------------------------------------------------
    def explain(
        self,
        user_idx: int,
        item_idx: int,
        hour: int = 12,
        dow: int = 2,
        mon: int = 5,
    ) -> Dict[str, Any]:
        contributions = self._model_contributions(user_idx, item_idx, hour, dow, mon)
        genre_match   = self._genre_match(user_idx, item_idx)
        similar_liked = self._similar_liked(user_idx, item_idx)

        explanation = dict(
            item_idx=item_idx,
            model_contributions=contributions,
            genre_match=genre_match,
            similar_liked=similar_liked,
            natural_language=self._nl_explanation(contributions, genre_match, similar_liked),
        )
        return explanation

    # ------------------------------------------------------------------
    def _model_contributions(self, user_idx, item_idx, hour, dow, mon) -> Dict:
        w_cf, w_cbf, w_ncf = self.weights
        cf_raw  = self.cf.predict(user_idx, item_idx) / 5.0
        cbf_raw = self.cbf.predict_score(user_idx, item_idx)
        ncf_raw = self.ncf.predict(user_idx, item_idx, hour, dow, mon)

        total = w_cf * cf_raw + w_cbf * cbf_raw + w_ncf * ncf_raw or 1e-9

        return {
            "collaborative_filtering": {
                "raw_score": round(cf_raw, 4),
                "weight": w_cf,
                "contribution": round(w_cf * cf_raw / total, 4),
            },
            "content_based": {
                "raw_score": round(cbf_raw, 4),
                "weight": w_cbf,
                "contribution": round(w_cbf * cbf_raw / total, 4),
            },
            "neural_cf": {
                "raw_score": round(ncf_raw, 4),
                "weight": w_ncf,
                "contribution": round(w_ncf * ncf_raw / total, 4),
            },
            "hybrid_score": round(total, 4),
        }

    def _genre_match(self, user_idx: int, item_idx: int) -> Dict[str, float]:
        """Fraction of user's liked items that belong to each genre of the target item."""
        item_row = self._item_row(item_idx)
        if item_row is None:
            return {}

        item_genres = [g for g in self.genre_cols if item_row.get(g, 0) == 1]
        liked_idxs  = self._user_liked.get(user_idx, [])
        total_liked = max(len(liked_idxs), 1)

        genre_counts: Dict[str, int] = {}
        for li in liked_idxs:
            lr = self._item_row(li)
            if lr is None:
                continue
            for g in item_genres:
                if lr.get(g, 0) == 1:
                    genre_counts[g] = genre_counts.get(g, 0) + 1

        return {g: round(genre_counts.get(g, 0) / total_liked, 4) for g in item_genres}

    def _similar_liked(self, user_idx: int, item_idx: int, top_k: int = 3) -> List[Dict]:
        """Items the user liked that share genres with the target item."""
        item_row = self._item_row(item_idx)
        if item_row is None:
            return []
        target_genres = frozenset(g for g in self.genre_cols if item_row.get(g, 0) == 1)

        liked_idxs = self._user_liked.get(user_idx, [])
        candidates = []
        for li in liked_idxs:
            if li == item_idx:
                continue
            lr = self._item_row(li)
            if lr is None:
                continue
            li_genres = frozenset(g for g in self.genre_cols if lr.get(g, 0) == 1)
            overlap = len(target_genres & li_genres)
            if overlap:
                candidates.append(
                    {"item_idx": li, "title": lr.get("title", "?"), "genre_overlap": overlap}
                )

        candidates.sort(key=lambda x: x["genre_overlap"], reverse=True)
        return candidates[:top_k]

    def _nl_explanation(self, contributions, genre_match, similar_liked) -> str:
        parts = []

        # Genre match
        if genre_match:
            top_genre = max(genre_match, key=genre_match.get)
            score = genre_match[top_genre]
            if score > 0.1:
                parts.append(f"matches your love of {top_genre} films")

        # Similar liked items
        if similar_liked:
            parts.append(f"is similar to '{similar_liked[0]['title']}' which you enjoyed")

        # Dominant model
        model_contribs = {k: v["contribution"] for k, v in contributions.items()
                         if k != "hybrid_score"}
        dominant = max(model_contribs, key=model_contribs.get)
        if dominant == "collaborative_filtering":
            parts.append("users with similar taste rate it highly")
        elif dominant == "neural_cf":
            parts.append("our neural model predicts a strong match")
        elif dominant == "content_based":
            parts.append("its content closely matches your viewing profile")

        if parts:
            return "Recommended because " + ", and ".join(parts) + "."
        return "Based on your overall viewing history and preferences."

    # ------------------------------------------------------------------
    def _item_row(self, item_idx: int) -> Optional[Dict]:
        mask = self.items_df["item_idx"] == item_idx
        if not mask.any():
            return None
        return self.items_df[mask].iloc[0].to_dict()


# ──────────────────────────────────────────────────────────────────────────────
class NcfShapExplainer:
    """
    SHAP DeepExplainer for the NCF Keras model.
    Requires: pip install shap tensorflow
    """

    def __init__(self, ncf: NeuralCollaborativeFilter, background_df: pd.DataFrame, n_bg: int = 100):
        import shap
        bg = background_df.sample(min(n_bg, len(background_df)), random_state=42)
        bg_inputs = [
            bg["user_idx"].values,
            bg["item_idx"].values,
            bg["hour"].values,
            bg["day_of_week"].values,
            bg["month"].values,
        ]
        self.explainer = shap.DeepExplainer(ncf.model, bg_inputs)
        self.feature_names = ["user_id", "item_id", "hour", "day_of_week", "month"]

    def explain(self, user_idx: int, item_idx: int, hour: int = 12, dow: int = 2, mon: int = 5) -> Dict:
        import shap
        sample = [
            np.array([[user_idx]]),
            np.array([[item_idx]]),
            np.array([[hour]]),
            np.array([[dow]]),
            np.array([[mon]]),
        ]
        shap_vals = self.explainer.shap_values(sample)
        # shap_vals is a list of arrays, one per input
        values = {name: float(np.squeeze(sv)) for name, sv in zip(self.feature_names, shap_vals)}
        return {"shap_values": values, "base_value": float(self.explainer.expected_value)}
