"""
Downloads MovieLens 100K and preprocesses it via the generic ingest pipeline.
Run once before training: python data/download.py
"""
import os
import sys
import urllib.request
import zipfile

# Allow importing ingest.py from model-server/utils/ and DatasetConfig from data/
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "model-server"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from dataset_config import DatasetConfig
from utils.ingest import process

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
RAW_DIR       = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

GENRE_COLS = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

MOVIELENS_CONFIG = DatasetConfig(
    dataset_name    = "movielens-100k",
    item_label      = "movie",
    feature_cols    = GENRE_COLS,
    feature_label   = "Genre",
    rating_threshold= 3.5,
    item_name_col   = "title",   # raw ML-100K column — ingest renames to "name"
)


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
    ratings, items, users = load_raw()
    process(
        ratings=ratings,
        items=items,
        config=MOVIELENS_CONFIG,
        output_dir=PROCESSED_DIR,
        users=users,
    )


if __name__ == "__main__":
    preprocess()
