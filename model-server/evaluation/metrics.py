"""Standard ranking metrics: Precision@K, Recall@K, NDCG@K, MRR."""
import numpy as np
from typing import List, Dict


def precision_at_k(recommended: List, relevant: set, k: int) -> float:
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: List, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: List, relevant: set, k: int) -> float:
    top_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, item in enumerate(top_k)
        if item in relevant
    )
    # Ideal DCG: all relevant items at top positions
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(recommended: List, relevant: set, k: int = None) -> float:
    """Reciprocal rank of the first relevant item, truncated at k if given."""
    top = recommended[:k] if k else recommended
    for rank, item in enumerate(top):
        if item in relevant:
            return 1.0 / (rank + 1)
    return 0.0


def evaluate_model(
    recommend_fn,
    test_df,
    train_df,
    k_values: List[int] = [5, 10, 20],
    n_users: int = None,
) -> Dict:
    """
    Evaluate a recommendation function over all test users.

    recommend_fn(user_idx) -> List[item_idx] (ranked, unseen)
    """
    # Ground truth: items each user interacted with in test set (rating >= 3.5)
    test_relevant = (
        test_df[test_df["rating"] >= 3.5]
        .groupby("user_idx")["item_idx"]
        .apply(set)
        .to_dict()
    )

    results = {k: {"precision": [], "recall": [], "ndcg": [], "mrr": []} for k in k_values}

    user_idxs = list(test_relevant.keys())
    if n_users:
        user_idxs = user_idxs[:n_users]

    for user_idx in user_idxs:
        relevant = test_relevant[user_idx]
        try:
            recommended = recommend_fn(user_idx)
        except Exception:
            continue

        for k in k_values:
            results[k]["precision"].append(precision_at_k(recommended, relevant, k))
            results[k]["recall"].append(recall_at_k(recommended, relevant, k))
            results[k]["ndcg"].append(ndcg_at_k(recommended, relevant, k))
            results[k]["mrr"].append(mrr(recommended, relevant, k))

    summary = {}
    for k in k_values:
        summary[f"precision@{k}"] = float(np.mean(results[k]["precision"]))
        summary[f"recall@{k}"] = float(np.mean(results[k]["recall"]))
        summary[f"ndcg@{k}"] = float(np.mean(results[k]["ndcg"]))
        summary[f"mrr@{k}"] = float(np.mean(results[k]["mrr"]))

    return summary
