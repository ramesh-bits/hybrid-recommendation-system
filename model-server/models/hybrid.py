"""
Hybrid Recommender – combines CF, CBF, and NCF via score-level fusion.

Scores from each model are min-max normalised to [0, 1] before fusion.
Weights are learned by grid-search on a validation set or can be set manually.
"""
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Set, Dict

from models.collaborative_filtering import SVDCollaborativeFilter
from models.content_based import ContentBasedFilter
from models.neural_cf import NeuralCollaborativeFilter
from models.context_aware import ContextAwareAdjuster


DEFAULT_WEIGHTS = (0.30, 0.30, 0.40)  # (CF, CBF, NCF)
SEARCH_GRID = [0.2, 0.3, 0.4, 0.5]


def _normalise(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return scores
    lo, hi = min(scores.values()), max(scores.values())
    span = hi - lo
    if span == 0:
        return {k: 0.5 for k in scores}
    return {k: (v - lo) / span for k, v in scores.items()}


class HybridRecommender:
    def __init__(
        self,
        cf: SVDCollaborativeFilter,
        cbf: ContentBasedFilter,
        ncf: NeuralCollaborativeFilter,
        context: Optional[ContextAwareAdjuster] = None,
        weights: Tuple[float, float, float] = DEFAULT_WEIGHTS,
    ):
        self.cf = cf
        self.cbf = cbf
        self.ncf = ncf
        self.context = context
        self.weights = weights

    # ------------------------------------------------------------------
    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        hour: int = 12,
        dow: int = 2,
        mon: int = 5,
        seen_items: Optional[Set[int]] = None,
        n_candidates: int = 200,
        items_df: Optional[pd.DataFrame] = None,
    ) -> List[Tuple[int, float]]:
        n_cand = max(n_candidates, n * 5)
        w_cf, w_cbf, w_ncf = self.weights

        cf_list  = self.cf.recommend(user_idx, n=n_cand, seen_items=seen_items)
        cbf_list = self.cbf.recommend(user_idx, n=n_cand, seen_items=seen_items)
        ncf_list = self.ncf.recommend(user_idx, n=n_cand, hour=hour, dow=dow, mon=mon, seen_items=seen_items)

        cf_norm  = _normalise(dict(cf_list))
        cbf_norm = _normalise(dict(cbf_list))
        ncf_norm = _normalise(dict(ncf_list))

        candidates = set(cf_norm) | set(cbf_norm) | set(ncf_norm)
        hybrid: Dict[int, float] = {}
        for it in candidates:
            hybrid[it] = (
                w_cf  * cf_norm.get(it,  0.0)
                + w_cbf * cbf_norm.get(it, 0.0)
                + w_ncf * ncf_norm.get(it, 0.0)
            )

        ranked = sorted(hybrid.items(), key=lambda x: x[1], reverse=True)

        if self.context and items_df is not None:
            ranked = self.context.adjust_scores(ranked, items_df, hour=hour, dow=dow)

        return ranked[:n]

    # ------------------------------------------------------------------
    def get_individual_scores(
        self,
        user_idx: int,
        item_idx: int,
        hour: int = 12,
        dow: int = 2,
        mon: int = 5,
    ) -> Dict[str, float]:
        return {
            "cf":  self.cf.predict(user_idx, item_idx),
            "cbf": self.cbf.predict_score(user_idx, item_idx),
            "ncf": self.ncf.predict(user_idx, item_idx, hour, dow, mon),
        }

    # ------------------------------------------------------------------
    def tune_weights(
        self,
        val_ratings: pd.DataFrame,
        train_ratings: pd.DataFrame,
        items_df: pd.DataFrame,
        k: int = 10,
        n_users: int = 50,
    ) -> Tuple[float, float, float]:
        """Grid-search hybrid weights on a validation set."""
        from evaluation.metrics import precision_at_k

        test_rel = (
            val_ratings[val_ratings["rating"] >= 3.5]
            .groupby("user_idx")["item_idx"]
            .apply(set)
            .to_dict()
        )
        seen_map = train_ratings.groupby("user_idx")["item_idx"].apply(set).to_dict()

        user_sample = list(test_rel.keys())[:n_users]
        best_weights, best_score = DEFAULT_WEIGHTS, -1.0

        combos = [
            (a, b, round(1 - a - b, 2))
            for a in SEARCH_GRID
            for b in SEARCH_GRID
            if round(1 - a - b, 2) >= 0.1
        ]

        for w_cf, w_cbf, w_ncf in combos:
            self.weights = (w_cf, w_cbf, w_ncf)
            scores = []
            for u in user_sample:
                recs = [it for it, _ in self.recommend(u, n=k, seen_items=seen_map.get(u))]
                scores.append(precision_at_k(recs, test_rel[u], k))
            avg = float(np.mean(scores))
            if avg > best_score:
                best_score = avg
                best_weights = (w_cf, w_cbf, w_ncf)

        self.weights = best_weights
        print(f"Best weights  CF={best_weights[0]}  CBF={best_weights[1]}  NCF={best_weights[2]}  P@{k}={best_score:.4f}")
        return best_weights

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        payload = {"weights": self.weights}
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load_weights(self, path: str) -> None:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.weights = payload["weights"]
