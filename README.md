# Hybrid Context-Aware Recommendation System with Explainability and Real-Time Adaptation

M.Tech. in Artificial Intelligence and Machine Learning — Dissertation Project
BITS Pilani

---

## What was built

A domain-agnostic hybrid recommender system. It ships pre-loaded with MovieLens 100K but any
dataset (e-commerce products, music, books, …) can be loaded at runtime via a CSV upload without
rebuilding the Docker images.

```
Code/
├── data/
│   ├── download.py              ← downloads & preprocesses MovieLens 100K (default dataset)
│   └── dataset_config.py        ← DatasetConfig Pydantic schema (single source of truth)
├── model-server/
│   ├── models/
│   │   ├── collaborative_filtering.py  ← SVD matrix factorisation (bias-adjusted)
│   │   ├── content_based.py            ← feature-vector cosine similarity + EMA profile update
│   │   ├── neural_cf.py                ← NCF (GMF ⊕ MLP) with hour/dow/month context inputs
│   │   ├── context_aware.py            ← per-feature temporal bias (hour, day-of-week)
│   │   └── hybrid.py                   ← min-max score fusion + grid-search weight tuning
│   ├── explainability/
│   │   └── explainer.py                ← model contributions, feature match, NL text, SHAP wrapper
│   ├── evaluation/
│   │   └── metrics.py                  ← Precision@K, Recall@K, NDCG@K, MRR
│   ├── utils/
│   │   ├── data_loader.py              ← loads processed pickles from /data/processed/
│   │   └── ingest.py                   ← generic CSV ingestion pipeline (any domain)
│   ├── train.py                        ← full training pipeline → /data/models/
│   └── main.py                         ← FastAPI model server (port 8001)
├── api/
│   ├── main.py                         ← FastAPI gateway (port 8000)
│   └── schemas.py                      ← Pydantic request/response schemas
├── ui/
│   └── app.py                          ← Streamlit dashboard (port 8501) — 6 pages
├── docker-compose.yml
├── Makefile
└── .env.example
```

---

## Architecture

Three microservices orchestrated with Docker Compose:

```
Browser
  └── Streamlit UI  :8501
        └── API Gateway  :8000
              └── Model Server  :8001
                    └── /data/  (shared volume — preprocessed data + model artefacts)
```

| Service | Port | Role |
|---------|------|------|
| `model-server` | 8001 | ML inference, training, explanation, dataset ingestion |
| `api` | 8000 | Public-facing FastAPI gateway |
| `ui` | 8501 | Streamlit dashboard |

---

## Models

| Component | Technique | File |
|-----------|-----------|------|
| Collaborative Filtering | Bias-adjusted SVD (scipy sparse) | `model-server/models/collaborative_filtering.py` |
| Content-Based Filtering | Feature-vector cosine similarity, EMA profile update | `model-server/models/content_based.py` |
| Neural CF | GMF ⊕ MLP with context embeddings (hour, day-of-week, month) | `model-server/models/neural_cf.py` |
| Context-Aware | Per-feature temporal bias adjustment | `model-server/models/context_aware.py` |
| Hybrid | Min-max normalised score fusion, grid-search weights | `model-server/models/hybrid.py` |
| Explainability | Model contributions, feature match, NL explanation, SHAP (NCF) | `model-server/explainability/explainer.py` |

---

## Getting Started

### With Docker (recommended)

```bash
# 1. Download and preprocess MovieLens 100K (~30 MB)
make data

# 2. Build images and start all three services
make up

# 3. Open the dashboard and click "Train models now"
#    Training takes ~5 minutes
open http://localhost:8501
```

API docs:
- Model Server: http://localhost:8001/docs
- API Gateway:  http://localhost:8000/docs

### Without Docker (local development)

```bash
make setup             # install Python deps into venv
make data              # download & preprocess MovieLens 100K
make train             # train all models

make local-model-server   # terminal 1  → :8001
make local-api            # terminal 2  → :8000
make local-ui             # terminal 3  → :8501
```

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| Recommendations | Top-K personalised recommendations with inline explanation cards |
| Model Comparison | Bar charts of Precision@K, Recall@K, NDCG@K across all four models |
| Explainability | Pie chart of model contributions, feature alignment, natural-language summary |
| Real-Time Simulation | Rate an item and watch the user profile and rankings update instantly |
| Data Explorer | User rating history, feature breakdown, dataset statistics |
| Dataset Upload | Upload a new domain's CSV files to replace the active dataset |

---

## Using a Custom Dataset

The system is domain-agnostic. To load your own data:

**Required files**

| File | Required columns |
|------|-----------------|
| `ratings.csv` | `user_id`, `item_id`, `rating`, `timestamp` (Unix seconds) |
| `items.csv` | `item_id`, your item-name column, binary `0/1` category columns |

**DatasetConfig JSON**

```json
{
  "dataset_name":     "amazon-electronics",
  "item_label":       "product",
  "feature_cols":     ["Computers", "Phones", "Cameras", "Audio"],
  "feature_label":    "Category",
  "rating_threshold": 4.0,
  "item_name_col":    "product_name"
}
```

**Option A — Dashboard**

Open the **Dataset Upload** page, drop in the two CSV files and the JSON config, then click
**Validate & Upload** and **Train models on new dataset**.

**Option B — API**

```bash
curl -X POST http://localhost:8001/upload \
  -F "ratings_file=@ratings.csv" \
  -F "items_file=@items.csv" \
  -F "config_json=$(cat config.json)"
```

**Option C — Makefile**

```bash
make ingest RATINGS=ratings.csv ITEMS=items.csv CONFIG=config.json
```

After ingestion, trigger training from the dashboard or via `POST /train`.

---

## Evaluation Metrics

Evaluated on a time-based held-out test set (last 20% of each user's interactions).

- **Precision@K** — fraction of top-K recommendations that are relevant
- **Recall@K** — fraction of relevant items found in top-K
- **NDCG@K** — ranking-quality-aware metric (Normalised Discounted Cumulative Gain)
- **MRR** — Mean Reciprocal Rank of the first relevant item

Results are saved to `data/models/eval_results.json` after training.

---

## Default Dataset (MovieLens 100K)

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) — downloaded automatically by `make data`.

| Attribute | Value |
|-----------|-------|
| Users | 943 |
| Movies | 1,682 |
| Ratings | 100,000 (scale 1–5) |
| Sparsity | ~93.7% |
| Time span | Sep 1997 – Apr 1998 |

---

## References

1. Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender Systems Handbook*. Springer.
2. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. *IEEE Computer*.
3. He, X., et al. (2017). Neural Collaborative Filtering. *WWW Conference*.
4. Rendle, S. (2010). Factorization Machines. *IEEE ICDM*.
5. Zhang, Y., & Chen, X. (2020). Explainable Recommendation: A Survey. *ACM Transactions*.
6. Adomavicius, G., & Tuzhilin, A. (2011). Context-Aware Recommender Systems. *ACM Transactions*.
