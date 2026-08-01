"""
Model-Server  —  FastAPI  (port 8001)

Loads pre-trained artefacts at startup and exposes ML inference endpoints
consumed by the API gateway. Training is triggered via POST /train.
New datasets are accepted via POST /upload.
"""
import os
import io
import json
import glob
import pickle
import subprocess
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
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

MODELS_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
EVAL_FILE     = os.path.join(MODELS_DIR, "eval_results.json")

app = FastAPI(title="Hybrid Recommender – Model Server", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Global state ─────────────────────────────────────────────────────────────
state: dict = {
    "cf": None, "cbf": None, "ncf": None, "ctx": None,
    "hybrid": None, "explainer": None,
    "ratings": None, "items": None, "users": None, "meta": None,
    "train": None, "test": None,
    "seen_map": None,
    "item_idx_to_id": None,
    "item_id_to_idx": None,
    "item_id_to_name": None,
    "loaded": False,
    "training": False,
    "last_trained": None,
}


def _p(filename: str) -> str:
    return os.path.join(MODELS_DIR, filename)


def _feature_cols(meta: dict):
    """Read feature_cols with backwards-compat fallback for old meta.pkl."""
    return meta.get("feature_cols") or meta.get("genre_cols", [])


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
    state["seen_map"]      = train.groupby("user_idx")["item_idx"].apply(set).to_dict()
    state["item_idx_to_id"]  = dict(zip(items["item_idx"], items["item_id"]))
    state["item_id_to_idx"]  = dict(zip(items["item_id"], items["item_idx"]))
    state["item_id_to_name"] = dict(zip(items["item_id"], items["name"]))

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
            feature_cols=_feature_cols(meta),
            item_label=meta.get("item_label", "item"),
            hybrid_weights=hybrid.weights,
        )

        state.update(cf=cf, cbf=cbf, ncf=ncf, ctx=ctx, hybrid=hybrid,
                     explainer=explainer, loaded=True)
        print("✓ All model artefacts loaded.")
    except FileNotFoundError:
        print("⚠  No model artefacts found — run POST /train first.")


def _invalidate_models():
    """Delete stale model artefacts after a new dataset is uploaded."""
    state["loaded"] = False
    state.update(cf=None, cbf=None, ncf=None, ctx=None, hybrid=None, explainer=None)
    for path in glob.glob(os.path.join(MODELS_DIR, "*.pkl")):
        os.remove(path)
    ncf_dir = os.path.join(MODELS_DIR, "ncf")
    if os.path.isdir(ncf_dir):
        import shutil
        shutil.rmtree(ncf_dir)
    if os.path.exists(EVAL_FILE):
        os.remove(EVAL_FILE)


@app.on_event("startup")
def startup():
    os.makedirs(MODELS_DIR, exist_ok=True)
    load_artefacts()


# ── Request models ────────────────────────────────────────────────────────────
class BehaviorContext(BaseModel):
    """Optional behavioural session context for a recommendation request:
    'the user just interacted with item X like this' — shifts the scoring
    for THIS request only, without persisting a profile update."""
    item_idx: int
    rating: Optional[float] = None
    view_time_sec: float = 0.0
    clicked: bool = False
    liked: bool = False
    wishlist: bool = False
    add_to_cart: bool = False
    ordered: bool = False


class RecommendRequest(BaseModel):
    user_idx: int
    n: int = 10
    hour: int = 12
    dow: int = 2
    mon: int = 5
    model: str = "hybrid"
    exclude_seen: bool = True
    behavior: Optional[BehaviorContext] = None


class ExplainRequest(BaseModel):
    user_idx: int
    item_idx: int
    hour: int = 12
    dow: int = 2
    mon: int = 5


class SimulateRequest(BaseModel):
    user_idx: int
    item_idx: int
    rating: Optional[float] = None       # explicit rating (1-5); optional if events given
    # Behavioural signals (all optional — simulated from the dashboard)
    view_time_sec: float = 0.0           # time spent viewing the item
    clicked: bool = False                # opened / clicked the item
    liked: bool = False                  # explicit like
    wishlist: bool = False               # added to wishlist / watch-later
    add_to_cart: bool = False            # added to cart
    ordered: bool = False                # purchased / fully consumed


# ── Helpers ───────────────────────────────────────────────────────────────────
def _engagement(view_time_sec: float, flags: dict) -> tuple:
    """(engagement_weight, effective_rating_if_none_given) from behavioural
    signals — same formula the ingestion pipeline uses for training weights."""
    from utils.ingest import (
        BEHAVIOR_FLAG_WEIGHTS, VIEW_TIME_SATURATION, VIEW_TIME_WEIGHT,
    )
    weight = 1.0
    for col, on in flags.items():
        if on:
            weight += BEHAVIOR_FLAG_WEIGHTS[col]
    if view_time_sec > 0:
        weight += VIEW_TIME_WEIGHT * min(view_time_sec / VIEW_TIME_SATURATION, 1.0)
    max_extra = sum(BEHAVIOR_FLAG_WEIGHTS.values()) + VIEW_TIME_WEIGHT
    inferred_rating = 3.0 + 2.0 * min((weight - 1.0) / max_extra, 1.0)
    return weight, inferred_rating


def _behavior_flags(req) -> dict:
    return {
        "clicked": req.clicked, "liked": req.liked, "wishlist": req.wishlist,
        "add_to_cart": req.add_to_cart, "ordered": req.ordered,
    }


def _native_id(value):
    """IDs may be ints (MovieLens) or strings (Amazon ASINs) — return a
    JSON-safe native value without assuming a type."""
    if value is None:
        return ""
    try:
        return int(value)          # numpy ints → int
    except (TypeError, ValueError):
        return str(value)


def _item_meta(item_idx: int) -> dict:
    item_id = state["item_idx_to_id"].get(item_idx)
    items   = state["items"]
    row     = items[items["item_idx"] == item_idx]
    if row.empty:
        return {"item_idx": int(item_idx), "item_id": _native_id(item_id), "name": "Unknown", "features": []}
    r = row.iloc[0]
    features = [f for f in _feature_cols(state["meta"]) if r.get(f, 0) == 1]
    return {"item_idx": int(item_idx), "item_id": _native_id(item_id), "name": str(r["name"]), "features": features}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": state["loaded"], "training": state["training"]}


@app.get("/dataset/info")
def dataset_info():
    meta = state.get("meta") or {}
    return {
        "dataset_name":  meta.get("dataset_name", "unknown"),
        "item_label":    meta.get("item_label", "item"),
        "feature_cols":  _feature_cols(meta),
        "feature_label": meta.get("feature_label", "Category"),
        "n_users":       meta.get("n_users", 0),
        "n_items":       meta.get("n_items", 0),
    }


@app.get("/schema")
def dataset_schema():
    """Expected upload schema + downloadable sample templates for the UI."""
    from utils.ingest import schema_description, sample_templates
    return {"schema": schema_description(), "templates": sample_templates()}


@app.post("/validate")
async def validate_dataset(
    ratings_file: UploadFile = File(...),
    items_file:   UploadFile = File(...),
    config_json:  str        = Form(...),
):
    """Dry-run validation: returns a structured report (errors, warnings,
    stats) WITHOUT ingesting anything or touching the active dataset."""
    import sys
    _DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    if _DATA_DIR not in sys.path:
        sys.path.insert(0, _DATA_DIR)
    from dataset_config import DatasetConfig
    from utils.ingest import validation_report

    try:
        config = DatasetConfig.model_validate_json(config_json)
    except Exception as e:
        return {"valid": False, "errors": [f"Invalid config JSON: {e}"], "warnings": [], "stats": {}}

    try:
        ratings_df = pd.read_csv(io.BytesIO(await ratings_file.read()))
        items_df   = pd.read_csv(io.BytesIO(await items_file.read()))
    except Exception as e:
        return {"valid": False, "errors": [f"Could not parse CSV: {e}"], "warnings": [], "stats": {}}

    return validation_report(ratings_df, items_df, config)


@app.post("/upload")
async def upload_dataset(
    ratings_file: UploadFile = File(...),
    items_file:   UploadFile = File(...),
    config_json:  str        = Form(...),
):
    """
    Upload a new dataset. Accepts ratings.csv, items.csv, and a DatasetConfig JSON.
    Invalidates any existing trained models — call POST /train after upload.
    """
    import sys
    _DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    if _DATA_DIR not in sys.path:
        sys.path.insert(0, _DATA_DIR)

    from dataset_config import DatasetConfig
    from utils.ingest import process as ingest_process

    try:
        config = DatasetConfig.model_validate_json(config_json)
    except Exception as e:
        raise HTTPException(422, f"Invalid config JSON: {e}")

    try:
        ratings_df = pd.read_csv(io.BytesIO(await ratings_file.read()))
        items_df   = pd.read_csv(io.BytesIO(await items_file.read()))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    # Invalidate stale models before writing new data
    current_name = (state.get("meta") or {}).get("dataset_name")
    if current_name != config.dataset_name:
        _invalidate_models()

    try:
        meta = ingest_process(
            ratings=ratings_df,
            items=items_df,
            config=config,
            output_dir=PROCESSED_DIR,
        )
    except ValueError as e:
        raise HTTPException(422, str(e))

    # Hot-reload data (models are invalidated — will be None until retrained)
    load_artefacts()

    return {
        "dataset_name": config.dataset_name,
        "n_users":      meta["n_users"],
        "n_items":      meta["n_items"],
        "feature_cols": meta["feature_cols"],
        "message":      "Dataset loaded. Call POST /train to train models on the new data.",
    }


class DemoDatasetRequest(BaseModel):
    category: str = "Software"          # Software | Musical_Instruments | Luxury_Beauty | ...
    max_ratings: Optional[int] = None   # cap most-recent reviews to bound training time


@app.post("/demo-dataset/prepare")
def prepare_demo_files(req: DemoDatasetRequest):
    """
    Download + convert an Amazon subset and RETURN the three upload files
    as text, WITHOUT ingesting. Lets the user download real CSVs from the
    dashboard and then demonstrate the manual upload → validate → train flow.
    """
    import sys, json as _json
    _DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    if _DATA_DIR not in sys.path:
        sys.path.insert(0, _DATA_DIR)
    import prepare_amazon

    cat = "".join(ch for ch in req.category if ch.isalnum() or ch == "_")
    if not cat:
        raise HTTPException(422, "Invalid category name.")
    try:
        ratings_df, items_df, config_dict = prepare_amazon.prepare(
            category=cat, max_ratings=req.max_ratings)
    except RuntimeError as e:
        raise HTTPException(502, str(e))

    return {
        "dataset_name": config_dict["dataset_name"],
        "n_ratings":    int(len(ratings_df)),
        "n_users":      int(ratings_df["user_id"].nunique()),
        "n_items":      int(items_df["item_id"].nunique()),
        "ratings_csv":  ratings_df.to_csv(index=False),
        "items_csv":    items_df.to_csv(index=False),
        "config_json":  _json.dumps(config_dict, indent=2),
    }


@app.post("/demo-dataset")
def load_demo_dataset(req: DemoDatasetRequest):
    """
    One-click demo: download a real public Amazon Reviews category subset
    (UCSD/McAuley 2018), convert it to the ingestion schema, validate it,
    and ingest it — entirely server-side, driven from the dashboard.
    """
    import sys
    _DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    if _DATA_DIR not in sys.path:
        sys.path.insert(0, _DATA_DIR)
    from dataset_config import DatasetConfig
    from utils.ingest import process as ingest_process, validation_report
    import prepare_amazon

    # basic input hygiene: category becomes part of a URL
    cat = "".join(ch for ch in req.category if ch.isalnum() or ch == "_")
    if not cat:
        raise HTTPException(422, "Invalid category name.")

    try:
        ratings_df, items_df, config_dict = prepare_amazon.prepare(
            category=cat, max_ratings=req.max_ratings)
    except RuntimeError as e:
        raise HTTPException(502, str(e))

    config = DatasetConfig(**config_dict)
    report = validation_report(ratings_df, items_df, config)
    if not report["valid"]:
        raise HTTPException(422, f"Prepared dataset failed validation: {report['errors']}")

    current_name = (state.get("meta") or {}).get("dataset_name")
    if current_name != config.dataset_name:
        _invalidate_models()

    meta = ingest_process(ratings=ratings_df, items=items_df,
                          config=config, output_dir=PROCESSED_DIR)
    load_artefacts()

    return {
        "dataset_name": config.dataset_name,
        "n_users":      meta["n_users"],
        "n_items":      meta["n_items"],
        "feature_cols": meta["feature_cols"],
        "validation":   report,
        "message":      "Demo dataset loaded. Call POST /train to train models on it.",
    }


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

    # Behavioural session context → temporary (non-persisted) profile shift
    profile_override = None
    behavior_info = None
    if req.behavior is not None:
        b = req.behavior
        weight, inferred = _engagement(b.view_time_sec, _behavior_flags(b))
        eff_rating = float(b.rating) if b.rating is not None else inferred
        profile_override = state["cbf"].preview_profile(
            req.user_idx, b.item_idx, eff_rating, engagement_weight=weight)
        # Don't recommend back the item the user just interacted with
        seen = set(seen) | {b.item_idx}
        behavior_info = {
            "item_idx": b.item_idx,
            "engagement_weight": round(weight, 3),
            "effective_rating": round(eff_rating, 2),
            "applied": profile_override is not None,
        }

    if req.model == "cf":
        raw = state["cf"].recommend(req.user_idx, n=n, seen_items=seen)
    elif req.model == "cbf":
        raw = state["cbf"].recommend(req.user_idx, n=n, seen_items=seen,
                                     profile_override=profile_override)
    elif req.model == "ncf":
        raw = state["ncf"].recommend(req.user_idx, n=n, hour=req.hour,
                                     dow=req.dow, mon=req.mon, seen_items=seen)
    else:
        raw = state["hybrid"].recommend(
            req.user_idx, n=n, hour=req.hour, dow=req.dow, mon=req.mon,
            seen_items=seen, items_df=state["items"],
            cbf_profile_override=profile_override,
        )

    results = []
    for rank, (item_idx, score) in enumerate(raw, 1):
        item = _item_meta(item_idx)
        item["rank"]  = rank
        item["score"] = round(float(score), 6)
        results.append(item)

    resp = {"user_idx": req.user_idx, "model": req.model, "recommendations": results}
    if behavior_info:
        resp["behavior_context"] = behavior_info
    return resp


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
    """
    Simulate a user interaction: an explicit rating and/or behavioural events
    (view time, click, like, wishlist, add-to-cart, order). Behavioural
    signals raise the engagement weight, which amplifies the real-time
    profile update — a purchase shifts the profile more than a lukewarm view.
    """
    if not state["loaded"]:
        raise HTTPException(503, "Models not loaded.")

    flags = _behavior_flags(req)
    weight, inferred = _engagement(req.view_time_sec, flags)
    # Effective rating: explicit if given, otherwise inferred from engagement
    # (weight 1.0 → neutral 3.0, max engagement → 5.0)
    effective_rating = float(req.rating) if req.rating is not None else inferred

    alpha = state["cbf"].update_profile(
        req.user_idx, req.item_idx, effective_rating, engagement_weight=weight
    )

    seen = state["seen_map"].setdefault(req.user_idx, set())
    seen.add(req.item_idx)

    return {
        "message": "Profile updated.",
        "user_idx": req.user_idx,
        "item_idx": req.item_idx,
        "engagement_weight": round(weight, 3),
        "effective_rating": round(effective_rating, 2),
        "profile_update_alpha": round(alpha, 4),
        "signals_applied": [k for k, v in flags.items() if v]
                           + (["view_time_sec"] if req.view_time_sec > 0 else []),
    }


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
        item = _item_meta(int(r["item_idx"]))
        item["rating"]    = float(r["rating"])
        item["timestamp"] = int(r["timestamp"])
        rows.append(item)
    return {"user_idx": user_idx, "history": rows}
