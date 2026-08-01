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
# Full weight simplex in steps of 0.1: any model may take any weight in
# [0, 1] (weights sum to 1). The previous grid capped NCF at 0.6 and forced
# CF/CBF to at least 0.2 each, which prevented the tuner from ever
# concentrating weight on the strongest component.
WEIGHT_STEP = 0.1
CTX_ALPHA_GRID = [0.0, 0.05, 0.15]   # context adjustment strength (0 = off)


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
        ctx_alpha: float = 0.15,
    ):
        self.cf = cf
        self.cbf = cbf
        self.ncf = ncf
        self.context = context
        self.weights = weights
        self.ctx_alpha = ctx_alpha

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
        cbf_profile_override=None,
    ) -> List[Tuple[int, float]]:
        n_cand = max(n_candidates, n * 5)
        w_cf, w_cbf, w_ncf = self.weights

        cf_list  = self.cf.recommend(user_idx, n=n_cand, seen_items=seen_items)
        cbf_list = self.cbf.recommend(user_idx, n=n_cand, seen_items=seen_items,
                                      profile_override=cbf_profile_override)
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

        if self.context and items_df is not None and self.ctx_alpha > 0:
            ranked = self.context.adjust_scores(
                ranked, items_df, hour=hour, dow=dow, alpha=self.ctx_alpha)

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
        n_users: int = 300,
    ) -> Tuple[float, float, float]:
        """
        Grid-search hybrid weights (full simplex, 0.1 steps) and the context
        adjustment strength on a validation set, maximising NDCG@k.

        Each component's candidate scores are computed ONCE per user and
        cached; the grid search then only re-weights the cached scores, so
        searching all 66 weight combinations x context strengths is cheap.
        """
        from evaluation.metrics import ndcg_at_k

        test_rel = (
            val_ratings[val_ratings["rating"] >= 3.5]
            .groupby("user_idx")["item_idx"]
            .apply(set)
            .to_dict()
        )
        seen_map = train_ratings.groupby("user_idx")["item_idx"].apply(set).to_dict()
        user_sample = list(test_rel.keys())[:n_users]

        # ── Cache per-user normalised component scores ────────────────────
        n_cand = 200
        cached = {}   # user → (cf_norm, cbf_norm, ncf_norm)
        for u in user_sample:
            seen = seen_map.get(u)
            cf_n  = _normalise(dict(self.cf.recommend(u, n=n_cand, seen_items=seen)))
            cbf_n = _normalise(dict(self.cbf.recommend(u, n=n_cand, seen_items=seen)))
            ncf_n = _normalise(dict(self.ncf.recommend(u, n=n_cand, seen_items=seen)))
            cached[u] = (cf_n, cbf_n, ncf_n)

        # ── Cache context adjustment per item (default serving context) ───
        ctx_score = {}
        if self.context is not None:
            feat_lookup = items_df.set_index("item_idx")[
                self.context.feature_cols].to_dict(orient="index")
            all_cands = set()
            for cf_n, cbf_n, ncf_n in cached.values():
                all_cands |= set(cf_n) | set(cbf_n) | set(ncf_n)
            for it in all_cands:
                ctx_score[it] = self.context.score(feat_lookup.get(it, {}), hour=12, dow=2)

        # ── Grid search over the weight simplex and context strength ──────
        steps = int(round(1.0 / WEIGHT_STEP))
        combos = [
            (round(i * WEIGHT_STEP, 2), round(j * WEIGHT_STEP, 2),
             round((steps - i - j) * WEIGHT_STEP, 2))
            for i in range(steps + 1)
            for j in range(steps + 1 - i)
        ]
        alphas = CTX_ALPHA_GRID if ctx_score else [0.0]

        best = (DEFAULT_WEIGHTS, 0.15, -1.0)
        for w_cf, w_cbf, w_ncf in combos:
            for alpha in alphas:
                scores = []
                for u in user_sample:
                    cf_n, cbf_n, ncf_n = cached[u]
                    cands = set(cf_n) | set(cbf_n) | set(ncf_n)
                    fused = {
                        it: (w_cf * cf_n.get(it, 0.0)
                             + w_cbf * cbf_n.get(it, 0.0)
                             + w_ncf * ncf_n.get(it, 0.0)
                             + alpha * ctx_score.get(it, 0.0))
                        for it in cands
                    }
                    recs = [it for it, _ in sorted(
                        fused.items(), key=lambda x: x[1], reverse=True)[:k]]
                    scores.append(ndcg_at_k(recs, test_rel[u], k))
                avg = float(np.mean(scores))
                if avg > best[2]:
                    best = ((w_cf, w_cbf, w_ncf), alpha, avg)

        self.weights, self.ctx_alpha = best[0], best[1]
        print(
            f"Best weights  CF={self.weights[0]}  CBF={self.weights[1]}  "
            f"NCF={self.weights[2]}  ctx_alpha={self.ctx_alpha}  "
            f"NDCG@{k}={best[2]:.4f}  (tuned on {len(user_sample)} users)"
        )
        return self.weights

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        payload = {"weights": self.weights, "ctx_alpha": self.ctx_alpha}
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load_weights(self, path: str) -> None:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.weights = payload["weights"]
        self.ctx_alpha = payload.get("ctx_alpha", 0.15)
