# CLAUDE.md — Hybrid Context-Aware Recommender System

M.Tech. AI & ML Dissertation, BITS Pilani.

## Project at a Glance

Hybrid recommender system combining CF, CBF, NCF, and context-awareness with XAI and real-time adaptation. Dataset: MovieLens 100K (943 users, 1,682 movies, 100 K ratings).

## Architecture

Three Docker microservices talking through a shared `/data/` volume:

```
Browser → Streamlit UI :8501 → API Gateway :8000 → Model Server :8001 → /data/
```

## Key Files

| File | Purpose |
|------|---------|
| `model-server/models/collaborative_filtering.py` | Bias-adjusted SVD (scipy sparse) |
| `model-server/models/content_based.py` | Genre-vector cosine similarity + EMA profile update |
| `model-server/models/neural_cf.py` | NCF: GMF ⊕ MLP with hour/dow/month context embeddings |
| `model-server/models/context_aware.py` | Per-genre temporal bias adjustment |
| `model-server/models/hybrid.py` | Min-max score fusion + grid-search weight tuning |
| `model-server/explainability/explainer.py` | Model contributions, genre match, NL explanation, SHAP |
| `model-server/evaluation/metrics.py` | Precision@K, Recall@K, NDCG@K, MRR |
| `model-server/train.py` | Full training pipeline → `data/models/` |
| `model-server/main.py` | Internal FastAPI model server (port 8001) |
| `api/main.py` | Public FastAPI gateway (port 8000) |
| `api/schemas.py` | Pydantic request/response schemas |
| `ui/app.py` | Streamlit dashboard — 5 pages (port 8501) |
| `data/download.py` | Auto-downloads and pickles MovieLens 100K |

## Run Order

```bash
make data          # download & preprocess MovieLens 100K
make up            # build Docker images and start all three services
# then open http://localhost:8501 and click "Train models now" (~5 min)
```

Local dev (no Docker):

```bash
make setup         # install Python deps into venv
make data
make train
make local-model-server   # terminal 1
make local-api            # terminal 2
make local-ui             # terminal 3
```

API docs: http://localhost:8001/docs (model server), http://localhost:8000/docs (gateway)

## Dashboard Pages

| Page | What it shows |
|------|--------------|
| Recommendations | Top-K personalised picks with explanation cards |
| Model Comparison | Precision@K, Recall@K, NDCG@K bar charts across all models |
| Explainability | Model contribution pie, genre alignment, NL summary |
| Real-Time Simulation | Rate a movie → watch profile and rankings update live |
| Data Explorer | Rating history, genre breakdown, dataset statistics |

## Evaluation

Time-based held-out test set (last 20% of each user's interactions). Results saved to `data/models/eval_results.json` after training.

## Stack

- Python / TensorFlow–Keras (NCF), scipy (SVD), scikit-learn
- FastAPI (two services), Streamlit (UI)
- Docker Compose for orchestration
- SHAP for model-level explainability
