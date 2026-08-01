"""
API Gateway  —  FastAPI  (port 8000)

Public-facing service. Proxies requests to the model-server and adds
lightweight request validation and response shaping.
"""
import os
import httpx
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    RecommendationResponse,
    ExplanationResponse,
    SimulateRequest,
    SimulateResponse,
    EvaluationResponse,
    DatasetInfoResponse,
)

MODEL_SERVER = os.getenv("MODEL_SERVER_URL", "http://model-server:8001")
TIMEOUT      = httpx.Timeout(60.0, connect=10.0)

app = FastAPI(title="Hybrid Recommender – API Gateway", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


async def _ms(method: str, path: str, **kwargs):
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await getattr(client, method)(f"{MODEL_SERVER}{path}", **kwargs)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    try:
        ms = await _ms("get", "/health")
    except Exception as e:
        ms = {"error": str(e)}
    return {"gateway": "ok", "model_server": ms, "timestamp": datetime.utcnow().isoformat()}


# ── Dataset ───────────────────────────────────────────────────────────────────
@app.get("/dataset/info")
async def dataset_info():
    return await _ms("get", "/dataset/info")


@app.get("/schema")
async def dataset_schema():
    """Expected dataset schema and sample CSV templates."""
    return await _ms("get", "/schema")


async def _forward_files(path: str, ratings_file, items_file, config_json):
    ratings_bytes = await ratings_file.read()
    items_bytes   = await items_file.read()
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        resp = await client.post(
            f"{MODEL_SERVER}{path}",
            files={
                "ratings_file": (ratings_file.filename, ratings_bytes, "text/csv"),
                "items_file":   (items_file.filename,   items_bytes,   "text/csv"),
            },
            data={"config_json": config_json},
        )
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


@app.post("/validate")
async def validate_dataset(
    ratings_file: UploadFile = File(...),
    items_file:   UploadFile = File(...),
    config_json:  str        = Form(...),
):
    """Dry-run schema/data validation — no ingestion, returns a report."""
    return await _forward_files("/validate", ratings_file, items_file, config_json)


@app.post("/upload")
async def upload_dataset(
    ratings_file: UploadFile = File(...),
    items_file:   UploadFile = File(...),
    config_json:  str        = Form(...),
):
    """Upload a new dataset (ratings.csv + items.csv + DatasetConfig JSON)."""
    return await _forward_files("/upload", ratings_file, items_file, config_json)


@app.post("/demo-dataset")
async def load_demo_dataset(payload: dict):
    """One-click server-side download + ingestion of an Amazon demo dataset."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
        resp = await client.post(f"{MODEL_SERVER}/demo-dataset", json=payload)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


@app.post("/demo-dataset/prepare")
async def prepare_demo_files(payload: dict):
    """Prepare the demo dataset's CSVs for download (no ingestion)."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
        resp = await client.post(f"{MODEL_SERVER}/demo-dataset/prepare", json=payload)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


@app.post("/train")
async def trigger_training():
    return await _ms("post", "/train")


# ── Recommendations ───────────────────────────────────────────────────────────
@app.get("/recommendations/{user_idx}")
async def get_recommendations(
    user_idx:     int,
    n:            int  = Query(10, ge=1, le=50),
    model:        str  = Query("hybrid", pattern="^(cf|cbf|ncf|hybrid)$"),
    hour:         int  = Query(12, ge=0, le=23),
    dow:          int  = Query(2,  ge=0, le=6),
    mon:          int  = Query(5,  ge=0, le=11),
    exclude_seen: bool = True,
    # Optional behavioural session context: "the user just interacted with
    # recent_item like this" — shifts scoring for this request only.
    recent_item:      Optional[int]   = Query(None, ge=0),
    recent_rating:    Optional[float] = Query(None, ge=1.0, le=5.0),
    recent_view_time: float = Query(0.0, ge=0.0),
    recent_clicked:   bool = False,
    recent_liked:     bool = False,
    recent_wishlist:  bool = False,
    recent_cart:      bool = False,
    recent_ordered:   bool = False,
):
    payload = dict(user_idx=user_idx, n=n, model=model,
                   hour=hour, dow=dow, mon=mon, exclude_seen=exclude_seen)
    if recent_item is not None:
        payload["behavior"] = dict(
            item_idx=recent_item, rating=recent_rating,
            view_time_sec=recent_view_time, clicked=recent_clicked,
            liked=recent_liked, wishlist=recent_wishlist,
            add_to_cart=recent_cart, ordered=recent_ordered,
        )
    data = await _ms("post", "/recommend", json=payload)
    return {**data, "context": {"hour": hour, "day_of_week": dow, "month": mon}}


# ── Explanation ───────────────────────────────────────────────────────────────
@app.get("/explain/{user_idx}/{item_idx}")
async def explain(
    user_idx: int,
    item_idx: int,
    hour: int = Query(12, ge=0, le=23),
    dow:  int = Query(2,  ge=0, le=6),
    mon:  int = Query(5,  ge=0, le=11),
):
    payload   = dict(user_idx=user_idx, item_idx=item_idx, hour=hour, dow=dow, mon=mon)
    data      = await _ms("post", "/explain", json=payload)
    item_data = await _ms("get", f"/items/{item_idx}")
    return {**data, "item_id": item_data.get("item_id"), "name": item_data.get("name")}


# ── Real-time simulation ──────────────────────────────────────────────────────
@app.post("/simulate")
async def simulate(req: SimulateRequest):
    sim_resp = await _ms("post", "/simulate", json=req.dict())
    rec_payload = dict(user_idx=req.user_idx, n=10, model="hybrid",
                       hour=12, dow=2, mon=5, exclude_seen=True)
    rec_resp = await _ms("post", "/recommend", json=rec_payload)
    return {**sim_resp, "updated_recommendations": rec_resp.get("recommendations", [])}


# ── User history ──────────────────────────────────────────────────────────────
@app.get("/users/{user_idx}/history")
async def user_history(user_idx: int, limit: int = Query(20, ge=1, le=100)):
    return await _ms("get", f"/user/{user_idx}/history?limit={limit}")


# ── Evaluation ────────────────────────────────────────────────────────────────
@app.get("/evaluation")
async def evaluation():
    return await _ms("get", "/evaluation")


# ── Item lookup ───────────────────────────────────────────────────────────────
@app.get("/items/{item_idx}")
async def get_item(item_idx: int):
    return await _ms("get", f"/items/{item_idx}")
