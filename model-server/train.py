"""
Full training pipeline.

Usage (inside model-server container or locally):
    python train.py           # full training (~15 epochs NCF)
    python train.py --quick   # fast demo run (5 epochs NCF)

Artefacts saved to  /data/models/
"""
import os
import sys
import time
import json
import argparse

# Allow imports from parent package when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.data_loader import load_meta, load_ratings, load_items, time_split
from models.collaborative_filtering import SVDCollaborativeFilter
from models.content_based import ContentBasedFilter
from models.neural_cf import NeuralCollaborativeFilter
from models.context_aware import ContextAwareAdjuster
from models.hybrid import HybridRecommender
from evaluation.metrics import evaluate_model

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "models")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Fast 5-epoch demo run")
    parser.add_argument("--tune-only", action="store_true",
                        help="Skip model training; load saved artefacts, re-tune hybrid weights, re-evaluate")
    args = parser.parse_args()
    ncf_epochs  = 5  if args.quick else 15
    eval_users  = 50 if args.quick else None   # None → evaluate ALL test users
    tune_users  = 30 if args.quick else 300

    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────
    print("Loading data …")
    meta     = load_meta()
    ratings  = load_ratings()
    items    = load_items()
    feature_cols = meta.get("feature_cols") or meta.get("genre_cols", [])
    n_users  = meta["n_users"]
    n_items  = meta["n_items"]

    # Chronological 80/20 test split per user, then a further per-user
    # chronological 90/10 train/validation split.
    # NOTE: this must be a per-user time split, NOT a row slice —
    # time_split returns rows grouped by user, so .iloc slicing would
    # remove the last ~10% of USERS from training entirely, leaving the
    # validation set full of users the models have never seen.
    train, test = time_split(ratings, test_ratio=0.2)
    train, val  = time_split(train, test_ratio=0.1)

    print(f"Train={len(train)}  Val={len(val)}  Test={len(test)}")

    seen_map = train.groupby("user_idx")["item_idx"].apply(set).to_dict()

    if args.tune_only:
        print("\n[1-4/5] --tune-only: loading saved model artefacts …")
        cf  = SVDCollaborativeFilter.load(os.path.join(MODELS_DIR, "cf_model.pkl"))
        cbf = ContentBasedFilter.load(os.path.join(MODELS_DIR, "cbf_model.pkl"))
        ncf = NeuralCollaborativeFilter.load(os.path.join(MODELS_DIR, "ncf"))
        ctx = ContextAwareAdjuster.load(os.path.join(MODELS_DIR, "ctx_model.pkl"))
    else:
        # ── Collaborative Filtering ────────────────────────────────────
        print("\n[1/5] Training SVD Collaborative Filter …")
        t0 = time.time()
        cf = SVDCollaborativeFilter(n_factors=50)
        cf.fit(train)
        cf.save(os.path.join(MODELS_DIR, "cf_model.pkl"))
        print(f"    done in {time.time()-t0:.1f}s")

        # ── Content-Based Filtering ────────────────────────────────────
        print("\n[2/5] Training Content-Based Filter …")
        t0 = time.time()
        cbf = ContentBasedFilter(feature_cols=feature_cols)
        cbf.fit(train, items)
        cbf.save(os.path.join(MODELS_DIR, "cbf_model.pkl"))
        print(f"    done in {time.time()-t0:.1f}s")

        # ── Neural Collaborative Filtering ────────────────────────────
        print("\n[3/5] Training NCF …")
        t0 = time.time()
        ncf = NeuralCollaborativeFilter(n_users=n_users, n_items=n_items, n_factors=32)
        ncf.fit(train, epochs=ncf_epochs, batch_size=512, neg_ratio=4)
        ncf.save(os.path.join(MODELS_DIR, "ncf"))
        print(f"    done in {time.time()-t0:.1f}s")

        # ── Context-Aware Adjuster ────────────────────────────────────
        print("\n[4/5] Fitting Context-Aware Adjuster …")
        ctx = ContextAwareAdjuster(feature_cols=feature_cols)
        ctx.fit(train, items)
        ctx.save(os.path.join(MODELS_DIR, "ctx_model.pkl"))

    # ── Hybrid: tune weights ──────────────────────────────────────────
    print("\n[5/5] Optimising hybrid weights …")
    hybrid = HybridRecommender(cf=cf, cbf=cbf, ncf=ncf, context=ctx)
    hybrid.tune_weights(val, train, items, k=10, n_users=tune_users)
    hybrid.save(os.path.join(MODELS_DIR, "hybrid_weights.pkl"))

    # ── Evaluation ───────────────────────────────────────────────────
    print("\nEvaluating on test set …")

    def make_rec_fn(model_name):
        def fn(user_idx):
            si = seen_map.get(user_idx, set())
            if model_name == "cf":
                return [it for it, _ in cf.recommend(user_idx, n=20, seen_items=si)]
            if model_name == "cbf":
                return [it for it, _ in cbf.recommend(user_idx, n=20, seen_items=si)]
            if model_name == "ncf":
                return [it for it, _ in ncf.recommend(user_idx, n=20, seen_items=si)]
            if model_name == "hybrid":
                # items_df enables the context-aware adjustment inside the
                # hybrid, so the evaluated system matches the served system
                return [it for it, _ in hybrid.recommend(
                    user_idx, n=20, seen_items=si, items_df=items)]
        return fn

    eval_results = {}
    for name in ["cf", "cbf", "ncf", "hybrid"]:
        print(f"  {name} …")
        eval_results[name] = evaluate_model(
            make_rec_fn(name), test, train, k_values=[5, 10, 20], n_users=eval_users
        )

    eval_results["_meta"] = {
        "hybrid_weights": {"cf": hybrid.weights[0], "cbf": hybrid.weights[1], "ncf": hybrid.weights[2]},
        "ctx_alpha": hybrid.ctx_alpha,
        "eval_users": "all" if eval_users is None else eval_users,
        "ncf_epochs": ncf_epochs,
    }

    with open(os.path.join(MODELS_DIR, "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=2)

    print("\n=== Evaluation Results ===")
    for model, metrics in eval_results.items():
        if model.startswith("_"):
            continue
        print(f"\n  {model.upper()}")
        for metric, val in metrics.items():
            print(f"    {metric}: {val:.4f}")
    print(f"\n  Tuned hybrid weights: {eval_results['_meta']['hybrid_weights']}")

    print(f"\nAll artefacts saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()
