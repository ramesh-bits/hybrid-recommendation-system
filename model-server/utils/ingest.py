"""
Generic dataset ingestion pipeline.

Accepts ratings.csv + items.csv + DatasetConfig and produces the six
pickle files consumed by the model-server:
  ratings.pkl, items.pkl, users.pkl, meta.pkl, user_encoder.pkl, item_encoder.pkl

Called by:
  - data/download.py  (host, MovieLens path)
  - model-server/main.py POST /upload  (inside Docker, any dataset)
"""
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Allow importing DatasetConfig when called from inside the container
# (data/ is mounted at /data, not on sys.path)
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
if _DATA_DIR not in sys.path:
    sys.path.insert(0, _DATA_DIR)

from dataset_config import DatasetConfig  # noqa: E402


REQUIRED_RATING_COLS = {"user_id", "item_id", "rating", "timestamp"}
REQUIRED_ITEM_COLS   = {"item_id"}

# Optional behavioural signal columns (all optional; any subset may be present).
# Binary flags are 0/1; view_time_sec is seconds spent viewing the item.
# Each signal adds its weight to the interaction confidence, which scales the
# interaction's influence during model training.
BEHAVIOR_FLAG_WEIGHTS = {
    "clicked":     0.3,   # user clicked / opened the item
    "liked":       0.5,   # explicit like
    "wishlist":    0.5,   # added to wishlist / watch-later
    "add_to_cart": 0.7,   # added to cart
    "ordered":     1.0,   # purchased / fully consumed
}
VIEW_TIME_COL       = "view_time_sec"
VIEW_TIME_SATURATION = 300.0   # seconds at which view time contributes fully
VIEW_TIME_WEIGHT     = 0.5
BEHAVIOR_COLS = list(BEHAVIOR_FLAG_WEIGHTS) + [VIEW_TIME_COL]


# ── Schema description (single source of truth for the UI) ───────────────────

def schema_description() -> dict:
    """Machine-readable schema of the expected upload files, rendered by the
    dashboard so users can see exactly what to provide before uploading."""
    return {
        "ratings": {
            "required": [
                {"column": "user_id",   "type": "int or string", "description": "Unique user identifier"},
                {"column": "item_id",   "type": "int or string", "description": "Unique item identifier"},
                {"column": "rating",    "type": "number",        "description": "Explicit rating (e.g. 1-5)"},
                {"column": "timestamp", "type": "int",           "description": "Interaction time as Unix seconds"},
            ],
            "optional": [
                {"column": "view_time_sec", "type": "number >= 0", "description": "Seconds spent viewing the item"},
                {"column": "clicked",       "type": "0/1", "description": "User clicked / opened the item"},
                {"column": "liked",         "type": "0/1", "description": "Explicit like"},
                {"column": "wishlist",      "type": "0/1", "description": "Added to wishlist / watch-later"},
                {"column": "add_to_cart",   "type": "0/1", "description": "Added to cart"},
                {"column": "ordered",       "type": "0/1", "description": "Purchased / fully consumed"},
            ],
        },
        "items": {
            "required": [
                {"column": "item_id", "type": "int or string", "description": "Must match item_id in ratings"},
                {"column": "<item_name_col>", "type": "string", "description": "Display name; set item_name_col in config"},
                {"column": "<feature columns>", "type": "0/1", "description": "One binary column per category, listed in config feature_cols"},
            ],
            "optional": [],
        },
        "config": [
            {"field": "dataset_name",     "type": "string", "description": "Unique slug, e.g. 'amazon-electronics'"},
            {"field": "item_label",       "type": "string", "description": "Singular noun: 'movie', 'product', 'song'"},
            {"field": "feature_cols",     "type": "list of strings", "description": "Binary category columns in items file"},
            {"field": "feature_label",    "type": "string", "description": "Display name: 'Genre', 'Category'"},
            {"field": "rating_threshold", "type": "number", "description": "Min rating counted as a positive interaction"},
            {"field": "item_name_col",    "type": "string", "description": "Items column holding the display name"},
        ],
    }


def sample_templates() -> dict:
    """Small example CSVs users can download as a starting point."""
    ratings = (
        "user_id,item_id,rating,timestamp,view_time_sec,clicked,add_to_cart,wishlist,ordered\n"
        "1,101,4.5,1719800000,240,1,1,0,1\n"
        "1,102,3.0,1719810000,35,1,0,0,0\n"
        "2,101,5.0,1719820000,410,1,1,1,1\n"
        "2,103,2.0,1719830000,10,0,0,0,0\n"
    )
    items = (
        "item_id,product_name,Computers,Phones,Cameras,Audio\n"
        "101,Wireless Headphones X2,0,0,0,1\n"
        "102,ProBook 14 Laptop,1,0,0,0\n"
        "103,SnapCam Mini,0,0,1,0\n"
    )
    config = (
        '{\n  "dataset_name": "my-dataset",\n  "item_label": "product",\n'
        '  "feature_cols": ["Computers", "Phones", "Cameras", "Audio"],\n'
        '  "feature_label": "Category",\n  "rating_threshold": 4.0,\n'
        '  "item_name_col": "product_name"\n}\n'
    )
    return {"ratings_csv": ratings, "items_csv": items, "config_json": config}


# ── Validation ────────────────────────────────────────────────────────────────

def _ratings_issues(df: pd.DataFrame):
    errors, warns = [], []
    missing = REQUIRED_RATING_COLS - set(df.columns)
    if missing:
        errors.append(
            f"ratings file is missing required column(s): {sorted(missing)}. "
            f"Required columns are: {sorted(REQUIRED_RATING_COLS)}."
        )
        return errors, warns   # further checks need the columns

    for col in ("rating", "timestamp"):
        coerced = pd.to_numeric(df[col], errors="coerce")
        n_bad = coerced.isna().sum() - df[col].isna().sum()
        if df[col].isna().sum():
            errors.append(f"Column '{col}' has {df[col].isna().sum()} empty value(s).")
        if n_bad:
            errors.append(
                f"Column '{col}' has {n_bad} non-numeric value(s) "
                f"(first bad row index: {int(coerced.isna().idxmax())})."
            )

    for col in ("user_id", "item_id"):
        n = df[col].isna().sum()
        if n:
            errors.append(f"Column '{col}' has {n} empty value(s); every row needs a {col}.")

    dupes = df.duplicated(subset=["user_id", "item_id", "timestamp"]).sum()
    if dupes:
        warns.append(f"{dupes} duplicate (user_id, item_id, timestamp) rows found — all will be kept.")

    for col in BEHAVIOR_FLAG_WEIGHTS:
        if col in df.columns:
            vals = set(pd.to_numeric(df[col], errors="coerce").dropna().unique()) - {0, 1, 0.0, 1.0}
            if vals:
                errors.append(
                    f"Behavioural column '{col}' must be binary 0/1; found values: {sorted(vals)[:5]}."
                )
    if VIEW_TIME_COL in df.columns:
        vt = pd.to_numeric(df[VIEW_TIME_COL], errors="coerce")
        if (vt.dropna() < 0).any():
            errors.append(f"'{VIEW_TIME_COL}' contains negative values; must be >= 0 seconds.")

    if len(df) < 100:
        warns.append(f"Only {len(df)} ratings — model quality may be low.")
    return errors, warns


def _items_issues(df: pd.DataFrame, config) -> tuple:
    errors, warns = [], []
    missing_base = REQUIRED_ITEM_COLS - set(df.columns)
    if missing_base:
        errors.append(f"items file is missing column(s): {sorted(missing_base)}")

    if config.item_name_col not in df.columns:
        errors.append(
            f"items file is missing the name column '{config.item_name_col}'. "
            f"Set item_name_col in your DatasetConfig to match the actual column name."
        )

    missing_feat = [c for c in config.feature_cols if c not in df.columns]
    if missing_feat:
        errors.append(f"items file is missing feature columns: {missing_feat}")

    for col in config.feature_cols:
        if col not in df.columns:
            continue
        bad_vals = set(df[col].dropna().unique()) - {0, 1, 0.0, 1.0}
        if bad_vals:
            warns.append(
                f"Feature column '{col}' contains non-binary values "
                f"{sorted(bad_vals)[:5]} — they will be coerced to 0/1."
            )
    return errors, warns


def validation_report(ratings: pd.DataFrame, items: pd.DataFrame, config) -> dict:
    """Structured pre-ingestion validation: errors (block ingestion),
    warnings (allowed but flagged), and dataset statistics."""
    r_err, r_warn = _ratings_issues(ratings)
    i_err, i_warn = _items_issues(items, config)

    stats = {}
    if not r_err:
        rated_items = set(ratings["item_id"])
        known_items = set(items["item_id"]) if "item_id" in items.columns else set()
        orphans = len(rated_items - known_items)
        if orphans:
            i_warn.append(
                f"{orphans} item_id(s) in ratings have no row in the items file — "
                "they will get empty feature vectors."
            )
        stats = {
            "n_ratings":  int(len(ratings)),
            "n_users":    int(ratings["user_id"].nunique()),
            "n_items":    int(ratings["item_id"].nunique()),
            "rating_min": float(pd.to_numeric(ratings["rating"], errors="coerce").min()),
            "rating_max": float(pd.to_numeric(ratings["rating"], errors="coerce").max()),
            "behavioural_columns_detected": [c for c in BEHAVIOR_COLS if c in ratings.columns],
        }

    return {
        "valid":    not (r_err + i_err),
        "errors":   r_err + i_err,
        "warnings": r_warn + i_warn,
        "stats":    stats,
    }


def validate_ratings(df: pd.DataFrame) -> None:
    errors, warns = _ratings_issues(df)
    for w in warns:
        warnings.warn(w)
    if errors:
        raise ValueError(" | ".join(errors))


def compute_interaction_weight(df: pd.DataFrame) -> pd.Series:
    """
    Confidence weight per interaction from optional behavioural signals.
    Base weight 1.0; each observed signal adds its contribution.
    Returns a Series of floats >= 1.0 (roughly in [1.0, 4.0]).
    """
    w = pd.Series(1.0, index=df.index)
    for col, add in BEHAVIOR_FLAG_WEIGHTS.items():
        if col in df.columns:
            w += add * pd.to_numeric(df[col], errors="coerce").fillna(0).clip(0, 1)
    if VIEW_TIME_COL in df.columns:
        vt = pd.to_numeric(df[VIEW_TIME_COL], errors="coerce").fillna(0).clip(lower=0)
        w += VIEW_TIME_WEIGHT * (vt / VIEW_TIME_SATURATION).clip(upper=1.0)
    return w


def validate_items(df: pd.DataFrame, config: DatasetConfig) -> None:
    errors, warns = _items_issues(df, config)
    for w in warns:
        warnings.warn(w)
    if errors:
        raise ValueError(" | ".join(errors))


# ── Core pipeline ─────────────────────────────────────────────────────────────

def process(
    ratings: pd.DataFrame,
    items: pd.DataFrame,
    config: DatasetConfig,
    output_dir: str,
    users: pd.DataFrame = None,
) -> dict:
    """
    Run the full ingestion pipeline. Returns the meta dict.

    Args:
        ratings:    DataFrame with columns user_id, item_id, rating, timestamp
        items:      DataFrame with item_id, config.item_name_col, + feature_cols
        config:     DatasetConfig describing the dataset
        output_dir: Directory to write pickle files into
        users:      Optional user metadata DataFrame (stub created if None)
    """
    validate_ratings(ratings)
    validate_items(items, config)

    os.makedirs(output_dir, exist_ok=True)

    # ── Encode users and items to 0-based integer indices ─────────────────
    user_enc = LabelEncoder().fit(ratings["user_id"])
    item_enc = LabelEncoder().fit(ratings["item_id"])

    ratings = ratings.copy()
    ratings["user_idx"] = user_enc.transform(ratings["user_id"])
    ratings["item_idx"] = item_enc.transform(ratings["item_id"])

    # ── Temporal context features ─────────────────────────────────────────
    ratings["datetime"]    = pd.to_datetime(ratings["timestamp"], unit="s")
    ratings["hour"]        = ratings["datetime"].dt.hour
    ratings["day_of_week"] = ratings["datetime"].dt.dayofweek
    ratings["month"]       = ratings["datetime"].dt.month - 1  # 0-based

    # ── Implicit positive label ───────────────────────────────────────────
    ratings["label"] = (ratings["rating"] >= config.rating_threshold).astype(np.float32)

    # ── Behavioural confidence weight (1.0 when no signals provided) ──────
    ratings["interaction_weight"] = compute_interaction_weight(ratings)

    # ── Items: align to item_idx, normalise name column, coerce features ──
    items = items.copy()
    # Only keep known item_ids (those that appear in ratings)
    items = items[items["item_id"].isin(item_enc.classes_)].copy()
    items["item_idx"] = item_enc.transform(items["item_id"])
    items = items.sort_values("item_idx").reset_index(drop=True)

    # Normalise the display name to a fixed internal column "name"
    if config.item_name_col != "name":
        items = items.rename(columns={config.item_name_col: "name"})

    # Coerce feature columns to integer 0/1
    for col in config.feature_cols:
        items[col] = items[col].fillna(0).clip(0, 1).astype(int)

    # ── Users: stub if not provided ───────────────────────────────────────
    if users is None:
        user_ids = pd.DataFrame({"user_id": user_enc.classes_})
        users = user_ids

    # ── Meta dict ─────────────────────────────────────────────────────────
    meta = config.to_meta_dict()
    meta.update({
        "n_users":           len(user_enc.classes_),
        "n_items":           len(item_enc.classes_),
        "user_enc_classes":  user_enc.classes_.tolist(),
        "item_enc_classes":  item_enc.classes_.tolist(),
        "behavior_cols_present": [c for c in BEHAVIOR_COLS if c in ratings.columns],
    })

    # ── Persist ───────────────────────────────────────────────────────────
    def _save_pkl(obj, name):
        path = os.path.join(output_dir, name)
        if isinstance(obj, pd.DataFrame):
            obj.to_pickle(path)
        else:
            with open(path, "wb") as f:
                pickle.dump(obj, f)

    _save_pkl(ratings, "ratings.pkl")
    _save_pkl(items,   "items.pkl")
    _save_pkl(users,   "users.pkl")
    _save_pkl(meta,    "meta.pkl")
    _save_pkl(user_enc, "user_encoder.pkl")
    _save_pkl(item_enc, "item_encoder.pkl")

    print(
        f"Ingested '{config.dataset_name}': "
        f"{meta['n_users']} users, {meta['n_items']} items, "
        f"{len(ratings)} ratings → {output_dir}"
    )
    return meta
