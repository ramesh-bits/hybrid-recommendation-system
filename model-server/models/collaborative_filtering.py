"""
SVD-based Collaborative Filtering using scipy sparse SVD.
Predicts explicit ratings (1-5) via bias-adjusted matrix factorization.
"""
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from typing import List, Tuple, Optional, Set


class SVDCollaborativeFilter:
    def __init__(self, n_factors: int = 50):
        self.n_factors = n_factors
        self.U: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None
        self.Vt: Optional[np.ndarray] = None
        self.global_mean: float = 0.0
        self.user_bias: dict = {}
        self.item_bias: dict = {}
        self.user_idx_to_row: dict = {}
        self.item_idx_to_col: dict = {}
        self.col_to_item_idx: dict = {}
        self._predicted: Optional[np.ndarray] = None  # full (n_users, n_items) matrix

    # ------------------------------------------------------------------
    def fit(self, ratings: pd.DataFrame) -> None:
        user_idxs = sorted(ratings["user_idx"].unique())
        item_idxs = sorted(ratings["item_idx"].unique())
        self.user_idx_to_row = {u: i for i, u in enumerate(user_idxs)}
        self.item_idx_to_col = {it: j for j, it in enumerate(item_idxs)}
        self.col_to_item_idx = {j: it for it, j in self.item_idx_to_col.items()}

        n_users = len(user_idxs)
        n_items = len(item_idxs)

        self.global_mean = ratings["rating"].mean()

        self.user_bias = (
            ratings.groupby("user_idx")["rating"].mean() - self.global_mean
        ).to_dict()
        self.item_bias = (
            ratings.groupby("item_idx")["rating"].mean() - self.global_mean
        ).to_dict()

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

        self._predicted = (
            latent
            + self.global_mean
            + user_bias_arr[:, None]
            + item_bias_arr[None, :]
        )
        self._predicted = np.clip(self._predicted, 1.0, 5.0)

    # ------------------------------------------------------------------
    def predict(self, user_idx: int, item_idx: int) -> float:
        r = self.user_idx_to_row.get(user_idx)
        c = self.item_idx_to_col.get(item_idx)
        if r is None or c is None:
            return float(self.global_mean)
        return float(self._predicted[r, c])

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        seen_items: Optional[Set[int]] = None,
    ) -> List[Tuple[int, float]]:
        r = self.user_idx_to_row.get(user_idx)
        if r is None:
            return []

        scores = self._predicted[r].copy()

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
