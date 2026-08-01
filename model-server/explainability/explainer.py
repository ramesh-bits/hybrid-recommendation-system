"""
Explainability engine for the hybrid recommender.

Produces four complementary explanation signals per (user, item) pair:

  1. model_contributions  – raw and normalised score from each sub-model
  2. feature_match        – how well the item's features align with the user's
                            historical feature preferences
  3. similar_liked        – items the user rated highly that share features
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
        feature_cols: List[str],
        item_label: str = "item",
        hybrid_weights=(0.30, 0.30, 0.40),
    ):
        self.cf           = cf
        self.cbf          = cbf
        self.ncf          = ncf
        self.items_df     = items_df
        self.ratings_df   = ratings_df
        self.feature_cols = feature_cols
        self.item_label   = item_label
        self.weights      = hybrid_weights

        self._user_liked: Dict[int, List[int]] = (
            ratings_df[ratings_df["rating"] >= 4.0]
            .groupby("user_idx")["item_idx"]
            .apply(list)
            .to_dict()
        )
        self._item_id_to_idx  = dict(zip(items_df["item_id"], items_df["item_idx"]))
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
        feature_match = self._feature_match(user_idx, item_idx)
        similar_liked = self._similar_liked(user_idx, item_idx)

        return dict(
            item_idx=item_idx,
            model_contributions=contributions,
            feature_match=feature_match,
            similar_liked=similar_liked,
            natural_language=self._nl_explanation(contributions, feature_match, similar_liked),
        )

    # ------------------------------------------------------------------
    def _model_contributions(self, user_idx, item_idx, hour, dow, mon) -> Dict:
        w_cf, w_cbf, w_ncf = self.weights
        cf_raw  = self.cf.predict(user_idx, item_idx) / 5.0
        cbf_raw = self.cbf.predict_score(user_idx, item_idx)
        ncf_raw = self.ncf.predict(user_idx, item_idx, hour, dow, mon)

        total = w_cf * cf_raw + w_cbf * cbf_raw + w_ncf * ncf_raw or 1e-9

        return {
            "collaborative_filtering": {
                "raw_score":    round(cf_raw, 4),
                "weight":       w_cf,
                "contribution": round(w_cf * cf_raw / total, 4),
            },
            "content_based": {
                "raw_score":    round(cbf_raw, 4),
                "weight":       w_cbf,
                "contribution": round(w_cbf * cbf_raw / total, 4),
            },
            "neural_cf": {
                "raw_score":    round(ncf_raw, 4),
                "weight":       w_ncf,
                "contribution": round(w_ncf * ncf_raw / total, 4),
            },
            "hybrid_score": round(total, 4),
        }

    def _feature_match(self, user_idx: int, item_idx: int) -> Dict[str, float]:
        """Fraction of user's liked items that share each feature of the target item."""
        item_row = self._item_row(item_idx)
        if item_row is None:
            return {}

        item_feats  = [f for f in self.feature_cols if item_row.get(f, 0) == 1]
        liked_idxs  = self._user_liked.get(user_idx, [])
        total_liked = max(len(liked_idxs), 1)

        feat_counts: Dict[str, int] = {}
        for li in liked_idxs:
            lr = self._item_row(li)
            if lr is None:
                continue
            for f in item_feats:
                if lr.get(f, 0) == 1:
                    feat_counts[f] = feat_counts.get(f, 0) + 1

        return {f: round(feat_counts.get(f, 0) / total_liked, 4) for f in item_feats}

    def _similar_liked(self, user_idx: int, item_idx: int, top_k: int = 3) -> List[Dict]:
        """Items the user liked that share features with the target item."""
        item_row = self._item_row(item_idx)
        if item_row is None:
            return []
        target_feats = frozenset(f for f in self.feature_cols if item_row.get(f, 0) == 1)

        candidates = []
        for li in self._user_liked.get(user_idx, []):
            if li == item_idx:
                continue
            lr = self._item_row(li)
            if lr is None:
                continue
            li_feats = frozenset(f for f in self.feature_cols if lr.get(f, 0) == 1)
            overlap  = len(target_feats & li_feats)
            if overlap:
                candidates.append({
                    "item_idx":     li,
                    "name":         lr.get("name", "?"),
                    "genre_overlap": overlap,
                })

        candidates.sort(key=lambda x: x["genre_overlap"], reverse=True)
        return candidates[:top_k]

    def _nl_explanation(self, contributions, feature_match, similar_liked) -> str:
        label  = self.item_label       # e.g. "movie", "product"
        labels = label + "s"           # plural

        parts = []

        if feature_match:
            top_feat = max(feature_match, key=feature_match.get)
            if feature_match[top_feat] > 0.1:
                parts.append(f"matches your love of {top_feat} {labels}")

        if similar_liked:
            parts.append(f"is similar to '{similar_liked[0]['name']}' which you enjoyed")

        model_contribs = {k: v["contribution"] for k, v in contributions.items()
                          if k != "hybrid_score"}
        dominant = max(model_contribs, key=model_contribs.get)
        if dominant == "collaborative_filtering":
            parts.append("users with similar taste rate it highly")
        elif dominant == "neural_cf":
            parts.append("our neural model predicts a strong match")
        elif dominant == "content_based":
            parts.append(f"its features closely match your {label} profile")

        if parts:
            return "Recommended because " + ", and ".join(parts) + "."
        return f"Based on your overall {label} history and preferences."

    # ------------------------------------------------------------------
    def _item_row(self, item_idx: int) -> Optional[Dict]:
        mask = self.items_df["item_idx"] == item_idx
        if not mask.any():
            return None
        return self.items_df[mask].iloc[0].to_dict()


# ──────────────────────────────────────────────────────────────────────────────
class NcfShapExplainer:
    """SHAP DeepExplainer for the NCF Keras model."""

    def __init__(self, ncf: NeuralCollaborativeFilter, background_df: pd.DataFrame, n_bg: int = 100):
        import shap
        bg = background_df.sample(min(n_bg, len(background_df)), random_state=42)
        bg_inputs = [
            bg["user_idx"].values, bg["item_idx"].values,
            bg["hour"].values, bg["day_of_week"].values, bg["month"].values,
        ]
        self.explainer     = shap.DeepExplainer(ncf.model, bg_inputs)
        self.feature_names = ["user_id", "item_id", "hour", "day_of_week", "month"]

    def explain(self, user_idx: int, item_idx: int, hour: int = 12, dow: int = 2, mon: int = 5) -> Dict:
        import shap
        sample = [
            np.array([[user_idx]]), np.array([[item_idx]]),
            np.array([[hour]]),     np.array([[dow]]),     np.array([[mon]]),
        ]
        shap_vals = self.explainer.shap_values(sample)
        values    = {name: float(np.squeeze(sv)) for name, sv in zip(self.feature_names, shap_vals)}
        return {"shap_values": values, "base_value": float(self.explainer.expected_value)}
