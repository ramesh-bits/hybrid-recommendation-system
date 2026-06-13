"""
API Gateway  —  FastAPI  (port 8000)

Public-facing service. Proxies requests to the model-server and adds
lightweight request validation, caching, and response shaping.
"""
import os
import httpx
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    RecommendationResponse,
    ExplanationResponse,
    SimulateRequest,
    SimulateResponse,
    EvaluationResponse,
)

MODEL_SERVER = os.getenv("MODEL_SERVER_URL", "http://model-server:8001")
TIMEOUT      = httpx.Timeout(60.0, connect=10.0)

app = FastAPI(title="Hybrid Recommender – API Gateway", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


async def _ms(method: str, path: str, **kwargs):
    """Call the model-server and raise on HTTP error."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await getattr(client, method)(f"{MODEL_SERVER}{path}", **kwargs)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


# ── Health ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    try:
        ms = await _ms("get", "/health")
    except Exception as e:
        ms = {"error": str(e)}
    return {
        "gateway": "ok",
        "model_server": ms,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/train")
async def trigger_training():
    return await _ms("post", "/train")


# ── Recommendations ───────────────────────────────────────────────────────────
@app.get("/recommendations/{user_idx}")
async def get_recommendations(
    user_idx: int,
    n: int          = Query(10, ge=1, le=50),
    model: str      = Query("hybrid", regex="^(cf|cbf|ncf|hybrid)$"),
    hour: int       = Query(12, ge=0, le=23),
    dow: int        = Query(2,  ge=0, le=6),
    mon: int        = Query(5,  ge=0, le=11),
    exclude_seen: bool = True,
):
    payload = dict(user_idx=user_idx, n=n, model=model,
                   hour=hour, dow=dow, mon=mon, exclude_seen=exclude_seen)
    data = await _ms("post", "/recommend", json=payload)
    return {
        **data,
        "context": {"hour": hour, "day_of_week": dow, "month": mon},
    }


# ── Explanation ───────────────────────────────────────────────────────────────
@app.get("/explain/{user_idx}/{item_idx}")
async def explain(
    user_idx: int,
    item_idx: int,
    hour: int = Query(12, ge=0, le=23),
    dow:  int = Query(2,  ge=0, le=6),
    mon:  int = Query(5,  ge=0, le=11),
):
    payload = dict(user_idx=user_idx, item_idx=item_idx, hour=hour, dow=dow, mon=mon)
    data = await _ms("post", "/explain", json=payload)

    # Enrich with item metadata
    item_data = await _ms("get", f"/items/{item_idx}")
    return {**data, "item_id": item_data.get("item_id"), "title": item_data.get("title")}


# ── Real-time simulation ─────────────────────────────────────────────────────
@app.post("/simulate")
async def simulate(req: SimulateRequest):
    sim_payload = req.dict()
    sim_resp = await _ms("post", "/simulate", json=sim_payload)

    # Return updated recommendations after adaptation
    rec_payload = dict(user_idx=req.user_idx, n=10, model="hybrid",
                       hour=12, dow=2, mon=5, exclude_seen=True)
    rec_resp = await _ms("post", "/recommend", json=rec_payload)

    return {
        **sim_resp,
        "updated_recommendations": rec_resp.get("recommendations", []),
    }


# ── User history ─────────────────────────────────────────────────────────────
@app.get("/users/{user_idx}/history")
async def user_history(user_idx: int, limit: int = Query(20, ge=1, le=100)):
    return await _ms("get", f"/user/{user_idx}/history?limit={limit}")


# ── Evaluation metrics ───────────────────────────────────────────────────────
@app.get("/evaluation")
async def evaluation():
    return await _ms("get", "/evaluation")


# ── Item lookup ──────────────────────────────────────────────────────────────
@app.get("/items/{item_idx}")
async def get_item(item_idx: int):
    return await _ms("get", f"/items/{item_idx}")
