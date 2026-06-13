"""
Model-Server  —  FastAPI  (port 8001)

Loads pre-trained artefacts at startup and exposes ML inference endpoints
consumed by the API gateway. Training is triggered via POST /train.
"""
import os
import json
import pickle
import subprocess
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.data_loader import load_meta, load_ratings, load_items, load_users
from models.collaborative_filtering import SVDCollaborativeFilter
from models.content_based import ContentBasedFilter
from models.neural_cf import NeuralCollaborativeFilter
from models.context_aware import ContextAwareAdjuster
from models.hybrid import HybridRecommender
from explainability.explainer import HybridExplainer
from evaluation.metrics import evaluate_model

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "models")
EVAL_FILE  = os.path.join(MODELS_DIR, "eval_results.json")

app = FastAPI(title="Hybrid Recommender – Model Server", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Global state ─────────────────────────────────────────────────────────────
state: dict = {
    "cf": None, "cbf": None, "ncf": None, "ctx": None,
    "hybrid": None, "explainer": None,
    "ratings": None, "items": None, "users": None, "meta": None,
    "train": None, "test": None,
    "seen_map": None,           # user_idx → set[item_idx]  (train)
    "item_idx_to_id": None,     # item_idx → item_id
    "item_id_to_idx": None,
    "item_id_to_title": None,
    "loaded": False,
    "training": False,
    "last_trained": None,
}


def _p(filename: str) -> str:
    return os.path.join(MODELS_DIR, filename)


def load_artefacts():
    from utils.data_loader import time_split

    meta    = load_meta()
    ratings = load_ratings()
    items   = load_items()
    users   = load_users()

    train, test = time_split(ratings, test_ratio=0.2)

    state["meta"]    = meta
    state["ratings"] = ratings
    state["items"]   = items
    state["users"]   = users
    state["train"]   = train
    state["test"]    = test
    state["seen_map"] = train.groupby("user_idx")["item_idx"].apply(set).to_dict()
    state["item_idx_to_id"] = dict(zip(items["item_idx"], items["item_id"]))
    state["item_id_to_idx"] = dict(zip(items["item_id"], items["item_idx"]))
    state["item_id_to_title"] = dict(zip(items["item_id"], items["title"]))

    try:
        cf  = SVDCollaborativeFilter.load(_p("cf_model.pkl"))
        cbf = ContentBasedFilter.load(_p("cbf_model.pkl"))
        ncf = NeuralCollaborativeFilter.load(_p("ncf"))
        ctx = ContextAwareAdjuster.load(_p("ctx_model.pkl"))

        hybrid = HybridRecommender(cf=cf, cbf=cbf, ncf=ncf, context=ctx)
        hybrid.load_weights(_p("hybrid_weights.pkl"))

        explainer = HybridExplainer(
            cf=cf, cbf=cbf, ncf=ncf,
            items_df=items, ratings_df=train,
            genre_cols=meta["genre_cols"],
            hybrid_weights=hybrid.weights,
        )

        state.update(cf=cf, cbf=cbf, ncf=ncf, ctx=ctx, hybrid=hybrid,
                     explainer=explainer, loaded=True)
        print("✓ All model artefacts loaded.")
    except FileNotFoundError:
        print("⚠  No model artefacts found — run POST /train first.")


@app.on_event("startup")
def startup():
    os.makedirs(MODELS_DIR, exist_ok=True)
    load_artefacts()


# ── Models ───────────────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    user_idx: int
    n: int = 10
    hour: int = 12
    dow: int = 2
    mon: int = 5
    model: str = "hybrid"          # cf | cbf | ncf | hybrid
    exclude_seen: bool = True


class ExplainRequest(BaseModel):
    user_idx: int
    item_idx: int
    hour: int = 12
    dow: int = 2
    mon: int = 5


class SimulateRequest(BaseModel):
    user_idx: int
    item_idx: int
    rating: float


# ── Helpers ───────────────────────────────────────────────────────────────────
def _item_meta(item_idx: int) -> dict:
    item_id = state["item_idx_to_id"].get(item_idx)
    items   = state["items"]
    row     = items[items["item_idx"] == item_idx]
    if row.empty:
        return {"item_idx": int(item_idx), "item_id": int(item_id or 0), "title": "Unknown", "genres": []}
    r = row.iloc[0]
    genres = [g for g in state["meta"]["genre_cols"] if r.get(g, 0) == 1]
    return {"item_idx": int(item_idx), "item_id": int(item_id or 0), "title": str(r["title"]), "genres": genres}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": state["loaded"], "training": state["training"]}


@app.post("/train")
def trigger_training(background: BackgroundTasks):
    if state["training"]:
        return {"message": "Training already in progress."}

    def _run():
        state["training"] = True
        try:
            subprocess.run(["python", "train.py"], cwd=os.path.dirname(__file__), check=True)
            load_artefacts()
            state["last_trained"] = datetime.utcnow().isoformat()
        finally:
            state["training"] = False

    background.add_task(_run)
    return {"message": "Training started in background."}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    if not state["loaded"]:
        raise HTTPException(503, "Models not loaded. Call POST /train first.")

    seen = state["seen_map"].get(req.user_idx, set()) if req.exclude_seen else set()
    n    = req.n

    if req.model == "cf":
        raw = state["cf"].recommend(req.user_idx, n=n, seen_items=seen)
    elif req.model == "cbf":
        raw = state["cbf"].recommend(req.user_idx, n=n, seen_items=seen)
    elif req.model == "ncf":
        raw = state["ncf"].recommend(req.user_idx, n=n, hour=req.hour,
                                     dow=req.dow, mon=req.mon, seen_items=seen)
    else:
        raw = state["hybrid"].recommend(
            req.user_idx, n=n, hour=req.hour, dow=req.dow, mon=req.mon,
            seen_items=seen, items_df=state["items"],
        )

    results = []
    for rank, (item_idx, score) in enumerate(raw, 1):
        meta = _item_meta(item_idx)
        meta["rank"]  = rank
        meta["score"] = round(float(score), 6)
        results.append(meta)

    return {"user_idx": req.user_idx, "model": req.model, "recommendations": results}


@app.post("/explain")
def explain(req: ExplainRequest):
    if not state["loaded"]:
        raise HTTPException(503, "Models not loaded.")
    exp = state["explainer"].explain(
        req.user_idx, req.item_idx, req.hour, req.dow, req.mon
    )
    return {"user_idx": req.user_idx, **exp}


@app.post("/simulate")
def simulate_interaction(req: SimulateRequest):
    """Real-time adaptation: update CBF user profile with a new rating."""
    if not state["loaded"]:
        raise HTTPException(503, "Models not loaded.")

    state["cbf"].update_profile(req.user_idx, req.item_idx, req.rating)

    # Update seen_map so updated recs exclude newly rated item
    seen = state["seen_map"].setdefault(req.user_idx, set())
    seen.add(req.item_idx)

    return {"message": "Profile updated.", "user_idx": req.user_idx, "item_idx": req.item_idx}


@app.get("/evaluation")
def get_evaluation():
    if os.path.exists(EVAL_FILE):
        with open(EVAL_FILE) as f:
            return json.load(f)
    return {}


@app.get("/users")
def list_users(limit: int = 20, offset: int = 0):
    users = state["users"]
    if users is None:
        raise HTTPException(503, "Data not loaded.")
    page = users.iloc[offset : offset + limit]
    return page.to_dict(orient="records")


@app.get("/items/{item_idx}")
def get_item(item_idx: int):
    return _item_meta(item_idx)


@app.get("/user/{user_idx}/history")
def user_history(user_idx: int, limit: int = 20):
    ratings = state["ratings"]
    if ratings is None:
        raise HTTPException(503, "Data not loaded.")
    hist = (
        ratings[ratings["user_idx"] == user_idx]
        .sort_values("timestamp", ascending=False)
        .head(limit)
    )
    rows = []
    for _, r in hist.iterrows():
        meta = _item_meta(int(r["item_idx"]))
        meta["rating"]    = float(r["rating"])
        meta["timestamp"] = int(r["timestamp"])
        rows.append(meta)
    return {"user_idx": user_idx, "history": rows}
