"""Loads preprocessed MovieLens data from disk. Shared by training and inference."""
import os
import pickle
import pandas as pd

PROCESSED_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "data", "processed",
)


def _path(filename: str) -> str:
    return os.path.join(PROCESSED_DIR, filename)


def load_meta() -> dict:
    with open(_path("meta.pkl"), "rb") as f:
        return pickle.load(f)


def load_ratings() -> pd.DataFrame:
    return pd.read_pickle(_path("ratings.pkl"))


def load_items() -> pd.DataFrame:
    return pd.read_pickle(_path("items.pkl"))


def load_users() -> pd.DataFrame:
    return pd.read_pickle(_path("users.pkl"))


def load_encoders():
    with open(_path("user_encoder.pkl"), "rb") as f:
        user_enc = pickle.load(f)
    with open(_path("item_encoder.pkl"), "rb") as f:
        item_enc = pickle.load(f)
    return user_enc, item_enc


def time_split(ratings: pd.DataFrame, test_ratio: float = 0.2):
    """Time-based train/test split: last `test_ratio` interactions per user go to test."""
    ratings = ratings.sort_values("timestamp")
    train_idx, test_idx = [], []
    for _, group in ratings.groupby("user_idx"):
        n = len(group)
        cutoff = int(n * (1 - test_ratio))
        train_idx.extend(group.index[:cutoff].tolist())
        test_idx.extend(group.index[cutoff:].tolist())
    return ratings.loc[train_idx].reset_index(drop=True), ratings.loc[test_idx].reset_index(drop=True)
