"""
Downloads MovieLens 100K and preprocesses it into pickle files consumed by the model server.
Run this once before training: python data/download.py
"""
import os
import urllib.request
import zipfile
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

GENRE_COLS = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def download():
    os.makedirs(RAW_DIR, exist_ok=True)
    zip_path = os.path.join(RAW_DIR, "ml-100k.zip")
    if not os.path.exists(os.path.join(RAW_DIR, "ml-100k")):
        if not os.path.exists(zip_path):
            print("Downloading MovieLens 100K …")
            urllib.request.urlretrieve(MOVIELENS_URL, zip_path)
        print("Extracting …")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(RAW_DIR)
    print("Raw data ready.")


def load_raw():
    base = os.path.join(RAW_DIR, "ml-100k")

    ratings = pd.read_csv(
        os.path.join(base, "u.data"),
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )

    item_cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRE_COLS
    items = pd.read_csv(
        os.path.join(base, "u.item"),
        sep="|",
        names=item_cols,
        encoding="latin-1",
    )

    users = pd.read_csv(
        os.path.join(base, "u.user"),
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"],
    )

    return ratings, items, users


def preprocess():
    download()
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    ratings, items, users = load_raw()

    # Consecutive 0-based indices for model embeddings
    user_enc = LabelEncoder().fit(ratings["user_id"])
    item_enc = LabelEncoder().fit(ratings["item_id"])
    ratings["user_idx"] = user_enc.transform(ratings["user_id"])
    ratings["item_idx"] = item_enc.transform(ratings["item_id"])

    # Temporal context features
    ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")
    ratings["hour"] = ratings["datetime"].dt.hour
    ratings["day_of_week"] = ratings["datetime"].dt.dayofweek
    ratings["month"] = ratings["datetime"].dt.month - 1  # 0-based

    # Item features aligned to item_idx order
    items["item_idx"] = item_enc.transform(items["item_id"].values)
    items = items.sort_values("item_idx").reset_index(drop=True)

    # Implicit feedback label (threshold 3.5)
    ratings["label"] = (ratings["rating"] >= 3.5).astype(np.float32)

    n_users = len(user_enc.classes_)
    n_items = len(item_enc.classes_)

    meta = {
        "n_users": n_users,
        "n_items": n_items,
        "genre_cols": GENRE_COLS,
        "user_enc_classes": user_enc.classes_.tolist(),
        "item_enc_classes": item_enc.classes_.tolist(),
    }

    # Persist
    ratings.to_pickle(os.path.join(PROCESSED_DIR, "ratings.pkl"))
    items.to_pickle(os.path.join(PROCESSED_DIR, "items.pkl"))
    users.to_pickle(os.path.join(PROCESSED_DIR, "users.pkl"))
    with open(os.path.join(PROCESSED_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    with open(os.path.join(PROCESSED_DIR, "user_encoder.pkl"), "wb") as f:
        pickle.dump(user_enc, f)
    with open(os.path.join(PROCESSED_DIR, "item_encoder.pkl"), "wb") as f:
        pickle.dump(item_enc, f)

    print(f"Done: {n_users} users, {n_items} items, {len(ratings)} ratings → {PROCESSED_DIR}")
    return ratings, items, users, meta


if __name__ == "__main__":
    preprocess()
