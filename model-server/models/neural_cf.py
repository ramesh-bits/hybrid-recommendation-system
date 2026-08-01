"""
Neural Collaborative Filtering (NCF) – He et al., WWW 2017.

Architecture
────────────
  GMF branch  : user_emb ⊙ item_emb   (element-wise product)
  MLP branch  : [user_emb ∥ item_emb ∥ ctx_emb] → Dense → … → Dense
  Output      : sigmoid( Dense(GMF ∥ MLP) )

Context inputs: hour (0-23), day_of_week (0-6), month (0-11).
Training: implicit feedback with random negative sampling.
"""
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from typing import List, Tuple, Optional, Set


class NeuralCollaborativeFilter:
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int = 32,
        mlp_layers: List[int] = [256, 128, 64, 32],
        dropout: float = 0.2,
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.mlp_layers = mlp_layers
        self.dropout = dropout
        self.model: Optional[keras.Model] = None
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        # ── Inputs ──────────────────────────────────────────────────────
        user_in = keras.Input(shape=(1,), name="user_in", dtype=tf.int32)
        item_in = keras.Input(shape=(1,), name="item_in", dtype=tf.int32)
        hour_in = keras.Input(shape=(1,), name="hour_in", dtype=tf.int32)
        dow_in  = keras.Input(shape=(1,), name="dow_in",  dtype=tf.int32)
        mon_in  = keras.Input(shape=(1,), name="mon_in",  dtype=tf.int32)

        # ── GMF branch ──────────────────────────────────────────────────
        user_gmf = keras.layers.Embedding(self.n_users, self.n_factors, name="user_gmf")(user_in)
        item_gmf = keras.layers.Embedding(self.n_items, self.n_factors, name="item_gmf")(item_in)
        user_gmf = keras.layers.Flatten()(user_gmf)
        item_gmf = keras.layers.Flatten()(item_gmf)
        gmf_out  = keras.layers.Multiply(name="gmf_out")([user_gmf, item_gmf])

        # ── MLP branch ──────────────────────────────────────────────────
        mlp_factor = self.mlp_layers[0] // 2
        user_mlp = keras.layers.Embedding(self.n_users, mlp_factor, name="user_mlp")(user_in)
        item_mlp = keras.layers.Embedding(self.n_items, mlp_factor, name="item_mlp")(item_in)
        user_mlp = keras.layers.Flatten()(user_mlp)
        item_mlp = keras.layers.Flatten()(item_mlp)

        # Context embeddings
        hour_emb = keras.layers.Embedding(24, 8, name="hour_emb")(hour_in)
        dow_emb  = keras.layers.Embedding(7,  4, name="dow_emb")(dow_in)
        mon_emb  = keras.layers.Embedding(12, 4, name="mon_emb")(mon_in)
        hour_emb = keras.layers.Flatten()(hour_emb)
        dow_emb  = keras.layers.Flatten()(dow_emb)
        mon_emb  = keras.layers.Flatten()(mon_emb)

        mlp_x = keras.layers.Concatenate(name="mlp_input")(
            [user_mlp, item_mlp, hour_emb, dow_emb, mon_emb]
        )
        for i, units in enumerate(self.mlp_layers):
            mlp_x = keras.layers.Dense(units, activation="relu", name=f"mlp_dense_{i}")(mlp_x)
            mlp_x = keras.layers.Dropout(self.dropout)(mlp_x)

        # ── Combine ─────────────────────────────────────────────────────
        combined = keras.layers.Concatenate(name="combine")([gmf_out, mlp_x])
        output   = keras.layers.Dense(1, activation="sigmoid", name="output")(combined)

        self.model = keras.Model(
            inputs=[user_in, item_in, hour_in, dow_in, mon_in],
            outputs=output,
        )
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

    # ------------------------------------------------------------------
    def _negative_sample(
        self, ratings: pd.DataFrame, neg_ratio: int = 4
    ) -> pd.DataFrame:
        all_item_set = set(range(self.n_items))
        user_seen = ratings.groupby("user_idx")["item_idx"].apply(set).to_dict()

        neg_rows = []
        for user_idx, group in ratings.groupby("user_idx"):
            seen = user_seen[user_idx]
            pool = list(all_item_set - seen)
            n_neg = min(neg_ratio * len(group), len(pool))
            neg_items = np.random.choice(pool, size=n_neg, replace=False)
            # Use median hour/dow/mon from user's history for negatives
            med_hour = int(group["hour"].median())
            med_dow  = int(group["day_of_week"].median())
            med_mon  = int(group["month"].median())
            for it in neg_items:
                neg_rows.append(
                    dict(user_idx=user_idx, item_idx=it, label=0.0,
                         hour=med_hour, day_of_week=med_dow, month=med_mon,
                         sample_weight=1.0)
                )

        neg_df = pd.DataFrame(neg_rows)
        pos_cols = ["user_idx", "item_idx", "label", "hour", "day_of_week", "month"]
        pos_df = ratings[pos_cols].copy()
        # Behavioural confidence: positives with strong engagement signals
        # (orders, cart adds, long view times) count more in the loss.
        if "interaction_weight" in ratings.columns:
            pos_df["sample_weight"] = ratings["interaction_weight"].fillna(1.0).values
        else:
            pos_df["sample_weight"] = 1.0
        combined = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1).reset_index(drop=True)
        return combined

    # ------------------------------------------------------------------
    def fit(
        self,
        ratings: pd.DataFrame,
        epochs: int = 15,
        batch_size: int = 512,
        neg_ratio: int = 4,
        validation_split: float = 0.1,
    ) -> keras.callbacks.History:
        df = self._negative_sample(ratings, neg_ratio)

        inputs = {
            "user_in": df["user_idx"].values,
            "item_in": df["item_idx"].values,
            "hour_in": df["hour"].values,
            "dow_in":  df["day_of_week"].values,
            "mon_in":  df["month"].values,
        }
        labels = df["label"].values.astype(np.float32)
        sample_weight = df["sample_weight"].values.astype(np.float32)

        callbacks = [
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        ]
        return self.model.fit(
            inputs, labels,
            sample_weight=sample_weight,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
        )

    # ------------------------------------------------------------------
    def _make_inputs(self, user_idx, item_idx, hour=12, dow=2, mon=5):
        return {
            "user_in": np.array([user_idx]),
            "item_in": np.array([item_idx]),
            "hour_in": np.array([hour]),
            "dow_in":  np.array([dow]),
            "mon_in":  np.array([mon]),
        }

    def predict(
        self, user_idx: int, item_idx: int, hour: int = 12, dow: int = 2, mon: int = 5
    ) -> float:
        inp = self._make_inputs(user_idx, item_idx, hour, dow, mon)
        return float(self.model(inp, training=False).numpy()[0, 0])

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        hour: int = 12,
        dow: int = 2,
        mon: int = 5,
        seen_items: Optional[Set[int]] = None,
    ) -> List[Tuple[int, float]]:
        all_items = np.arange(self.n_items)
        n_items = len(all_items)
        batch = {
            "user_in": np.full(n_items, user_idx),
            "item_in": all_items,
            "hour_in": np.full(n_items, hour),
            "dow_in":  np.full(n_items, dow),
            "mon_in":  np.full(n_items, mon),
        }
        scores = self.model.predict(batch, batch_size=1024, verbose=0).flatten()

        if seen_items:
            for it in seen_items:
                if 0 <= it < n_items:
                    scores[it] = -1.0

        top_k = np.argpartition(scores, -n)[-n:]
        top_k = top_k[np.argsort(scores[top_k])[::-1]]
        return [(int(it), float(scores[it])) for it in top_k]

    # ------------------------------------------------------------------
    def save(self, model_dir: str) -> None:
        os.makedirs(model_dir, exist_ok=True)
        self.model.save(os.path.join(model_dir, "ncf_model.keras"))
        meta = dict(
            n_users=self.n_users, n_items=self.n_items,
            n_factors=self.n_factors, mlp_layers=self.mlp_layers, dropout=self.dropout,
        )
        with open(os.path.join(model_dir, "ncf_meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

    @staticmethod
    def load(model_dir: str) -> "NeuralCollaborativeFilter":
        with open(os.path.join(model_dir, "ncf_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        obj = NeuralCollaborativeFilter.__new__(NeuralCollaborativeFilter)
        obj.__dict__.update(meta)
        obj.model = keras.models.load_model(os.path.join(model_dir, "ncf_model.keras"))
        return obj
