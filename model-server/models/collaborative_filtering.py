"""
SVD-based Collaborative Filtering using scipy sparse SVD.
Predicts explicit ratings (1-5) via bias-adjusted matrix factorization.

Ranking quality fixes (v2):
  - User/item biases are regularised with shrinkage (bias_reg): items or
    users with few ratings are pulled towards the global mean, so an item
    rated 5 stars once no longer dominates the recommendation list.
  - Ranking scores are NOT clipped to [1, 5]. Clipping created large
    groups of items tied at exactly 5.0, making the ranking among them
    arbitrary. Clipping is applied only in predict() for display purposes.
  - An optional log-popularity prior (pop_beta) is added to the ranking
    score, which counteracts the tendency of rating-prediction models to
    surface obscure items when used for top-K ranking.
  - Optional per-interaction confidence weights (interaction_weight
    column) scale each rating's contribution to the bias estimates.
"""
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from typing import List, Tuple, Optional, Set


class SVDCollaborativeFilter:
    def __init__(self, n_factors: int = 50, bias_reg: float = 25.0, pop_beta: float = 0.5):
        self.n_factors = n_factors
        self.bias_reg = bias_reg
        self.pop_beta = pop_beta
        self.U: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None
        self.Vt: Optional[np.ndarray] = None
        self.global_mean: float = 0.0
        self.user_bias: dict = {}
        self.item_bias: dict = {}
        self.user_idx_to_row: dict = {}
        self.item_idx_to_col: dict = {}
        self.col_to_item_idx: dict = {}
        self.item_pop_prior: dict = {}
        self._predicted: Optional[np.ndarray] = None  # full (n_users, n_items) matrix
        self._ranking: Optional[np.ndarray] = None    # predicted + popularity prior

    def __setstate__(self, state):
        # Migrate pickles saved before the ranking fixes
        state.setdefault("bias_reg", 25.0)
        state.setdefault("pop_beta", 0.5)
        state.setdefault("item_pop_prior", {})
        if "_ranking" not in state and state.get("_predicted") is not None:
            state["_ranking"] = state["_predicted"]
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    def fit(self, ratings: pd.DataFrame) -> None:
        user_idxs = sorted(ratings["user_idx"].unique())
        item_idxs = sorted(ratings["item_idx"].unique())
        self.user_idx_to_row = {u: i for i, u in enumerate(user_idxs)}
        self.item_idx_to_col = {it: j for j, it in enumerate(item_idxs)}
        self.col_to_item_idx = {j: it for it, j in self.item_idx_to_col.items()}

        n_users = len(user_idxs)
        n_items = len(item_idxs)

        # Optional per-interaction confidence weights (behavioural signals)
        if "interaction_weight" in ratings.columns:
            w = ratings["interaction_weight"].fillna(1.0).clip(lower=0.1).values
        else:
            w = np.ones(len(ratings))
        ratings = ratings.assign(_w=w, _wr=w * ratings["rating"].values)

        self.global_mean = ratings["_wr"].sum() / ratings["_w"].sum()

        # Regularised (shrunk) biases:  b = Σw·(r − μ) / (λ + Σw)
        lam = self.bias_reg
        ustats = ratings.groupby("user_idx").agg(wr=("_wr", "sum"), w=("_w", "sum"))
        self.user_bias = (
            (ustats["wr"] - self.global_mean * ustats["w"]) / (lam + ustats["w"])
        ).to_dict()
        istats = ratings.groupby("item_idx").agg(wr=("_wr", "sum"), w=("_w", "sum"))
        self.item_bias = (
            (istats["wr"] - self.global_mean * istats["w"]) / (lam + istats["w"])
        ).to_dict()

        # Log-popularity prior per item (weighted interaction counts)
        pop = ratings.groupby("item_idx")["_w"].sum()
        self.item_pop_prior = {it: float(np.log1p(c)) for it, c in pop.items()}

        # Centre ratings by subtracting global mean + biases
        rows = ratings["user_idx"].map(self.user_idx_to_row).values
        cols = ratings["item_idx"].map(self.item_idx_to_col).values
        data = (
            ratings["rating"].values
            - self.global_mean
            - ratings["user_idx"].map(self.user_bias).fillna(0).values
            - ratings["item_idx"].map(self.item_bias).fillna(0).values
        )

        R = csr_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)

        k = min(self.n_factors, min(n_users, n_items) - 1)
        U, s, Vt = svds(R, k=k)

        # Sort by descending singular value
        order = np.argsort(s)[::-1]
        self.U = U[:, order]
        self.sigma = np.diag(s[order])
        self.Vt = Vt[order, :]

        # Pre-compute full prediction matrix for fast batch recommend
        latent = self.U @ self.sigma @ self.Vt  # (n_users, n_items)

        user_bias_arr = np.array([self.user_bias.get(u, 0.0) for u in user_idxs])
        item_bias_arr = np.array([self.item_bias.get(it, 0.0) for it in item_idxs])

        # Unclipped rating estimate — used for ranking
        self._predicted = (
            latent
            + self.global_mean
            + user_bias_arr[:, None]
            + item_bias_arr[None, :]
        )

        # Ranking score adds a log-popularity prior to counteract the
        # obscure-item bias of pure rating prediction in top-K ranking.
        pop_arr = np.array([self.item_pop_prior.get(it, 0.0) for it in item_idxs])
        self._ranking = self._predicted + self.pop_beta * pop_arr[None, :]

    # ------------------------------------------------------------------
    def predict(self, user_idx: int, item_idx: int) -> float:
        r = self.user_idx_to_row.get(user_idx)
        c = self.item_idx_to_col.get(item_idx)
        if r is None or c is None:
            return float(self.global_mean)
        # Clip only for display: keeps the value on the 1-5 rating scale
        return float(np.clip(self._predicted[r, c], 1.0, 5.0))

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        seen_items: Optional[Set[int]] = None,
    ) -> List[Tuple[int, float]]:
        r = self.user_idx_to_row.get(user_idx)
        if r is None:
            return []

        scores = self._ranking[r].copy()

        if seen_items:
            for it in seen_items:
                c = self.item_idx_to_col.get(it)
                if c is not None:
                    scores[c] = -np.inf

        top_cols = np.argpartition(scores, -n)[-n:]
        top_cols = top_cols[np.argsort(scores[top_cols])[::-1]]
        return [(self.col_to_item_idx[c], float(scores[c])) for c in top_cols]

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "SVDCollaborativeFilter":
        with open(path, "rb") as f:
            return pickle.load(f)
