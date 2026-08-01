"""
Prepare an Amazon Reviews (2018, UCSD/McAuley) category subset for upload
into the recommender via the Dataset Upload page.

Produces three files matching the system's ingestion schema:
    ratings.csv   user_id, item_id, rating, timestamp, ordered
    items.csv     item_id, product_name, <binary subcategory columns>
    config.json   DatasetConfig for the dashboard upload

The real `verified` purchase flag is mapped to the behavioural column
`ordered`, demonstrating implicit-signal ingestion on real data.

Usage (from the repo root, venv active):
    python data/prepare_amazon.py                          # Software (~13K reviews, fast)
    python data/prepare_amazon.py --category Musical_Instruments --max-ratings 50000

Dataset citation:
    Ni, Li, McAuley. "Justifying Recommendations using Distantly-Labeled
    Reviews and Fine-Grained Aspects." EMNLP 2019.
"""
import argparse
import gzip
import json
import os
import urllib.request
from collections import Counter

import pandas as pd

BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2"
REVIEW_URL = BASE + "/categoryFilesSmall/{cat}_5.json.gz"
META_URLS = [
    BASE + "/metaFiles2/meta_{cat}.json.gz",
    BASE + "/metaFiles/meta_{cat}.json.gz",
]
N_FEATURES = 12          # number of subcategory columns to keep


def download(url: str, dest: str) -> bool:
    if os.path.exists(dest):
        print(f"  already downloaded: {dest}")
        return True
    print(f"  downloading {url} …")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            f.write(r.read())
        return True
    except Exception as e:
        print(f"    failed: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def read_jsonl_gz(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def prepare(category: str = "Software", max_ratings: int = None, raw_dir: str = None):
    """
    Download and convert an Amazon category subset.
    Returns (ratings_df, items_df, config_dict).
    Callable from the CLI below or from the model-server /demo-dataset endpoint.
    """
    cat = category
    raw = raw_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
    os.makedirs(raw, exist_ok=True)

    # ── Download ──────────────────────────────────────────────────────────
    print(f"[1/4] Downloading '{cat}' …")
    rev_gz = os.path.join(raw, f"{cat}_5.json.gz")
    if not download(REVIEW_URL.format(cat=cat), rev_gz):
        raise RuntimeError(
            f"Could not download the reviews file for '{cat}' — check the "
            "category name and the internet connection.")
    meta_gz = os.path.join(raw, f"meta_{cat}.json.gz")
    if not any(download(u.format(cat=cat), meta_gz) for u in META_URLS):
        raise RuntimeError(f"Could not download the metadata file for '{cat}'.")

    # ── Reviews → ratings ─────────────────────────────────────────────────
    print("[2/4] Parsing reviews …")
    rows = []
    for r in read_jsonl_gz(rev_gz):
        if not all(k in r for k in ("reviewerID", "asin", "overall", "unixReviewTime")):
            continue
        rows.append({
            "user_id":   r["reviewerID"],
            "item_id":   r["asin"],
            "rating":    float(r["overall"]),
            "timestamp": int(r["unixReviewTime"]),
            "ordered":   int(bool(r.get("verified", False))),   # real behavioural signal
        })
    ratings = pd.DataFrame(rows).drop_duplicates(
        subset=["user_id", "item_id", "timestamp"])
    if max_ratings and len(ratings) > max_ratings:
        ratings = ratings.nlargest(max_ratings, "timestamp")
        # keep the dataset 3-core after sampling so CF has signal
        for _ in range(3):
            vc_u = ratings["user_id"].value_counts()
            vc_i = ratings["item_id"].value_counts()
            ratings = ratings[
                ratings["user_id"].isin(vc_u[vc_u >= 3].index)
                & ratings["item_id"].isin(vc_i[vc_i >= 3].index)
            ]
    print(f"  {len(ratings):,} ratings, {ratings['user_id'].nunique():,} users, "
          f"{ratings['item_id'].nunique():,} products")

    # ── Metadata → items ──────────────────────────────────────────────────
    print("[3/4] Parsing product metadata …")
    keep = set(ratings["item_id"])
    names, cats = {}, {}
    for m in read_jsonl_gz(meta_gz):
        asin = m.get("asin")
        if asin not in keep:
            continue
        title = (m.get("title") or "").strip()
        names[asin] = title[:120] if title else asin
        cl = m.get("category") or m.get("categories") or []
        if cl and isinstance(cl[0], list):        # old nested format
            cl = cl[0]
        # skip the root category (equal to the subset name), keep subcategories
        cats[asin] = [c.strip() for c in cl[1:] if c and len(c.strip()) > 1]

    top = [c for c, _ in Counter(
        c for lst in cats.values() for c in lst).most_common(N_FEATURES)]
    if not top:
        raise RuntimeError("No subcategories found in metadata — try another category.")
    print(f"  feature columns: {top}")

    items = pd.DataFrame({"item_id": sorted(keep)})
    items["product_name"] = items["item_id"].map(lambda a: names.get(a, a))
    for c in top:
        items[c] = items["item_id"].map(lambda a, c=c: int(c in cats.get(a, [])))

    config = {
        "dataset_name":     f"amazon-{cat.lower().replace('_','-')}",
        "item_label":       "product",
        "feature_cols":     top,
        "feature_label":    "Category",
        "rating_threshold": 4.0,
        "item_name_col":    "product_name",
    }
    return ratings, items, config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Software",
                    help="Amazon category, e.g. Software, Musical_Instruments, Luxury_Beauty")
    ap.add_argument("--max-ratings", type=int, default=None,
                    help="Keep only the most recent N reviews (caps training time)")
    ap.add_argument("--out", default=None, help="Output directory")
    args = ap.parse_args()

    ratings, items, config = prepare(args.category, args.max_ratings)

    out = args.out or os.path.join(
        os.path.dirname(__file__), "uploads_demo", args.category.lower())
    os.makedirs(out, exist_ok=True)
    print("[4/4] Writing upload files …")
    ratings.to_csv(os.path.join(out, "ratings.csv"), index=False)
    items.to_csv(os.path.join(out, "items.csv"), index=False)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone → {out}")
    print("Next: open the dashboard → Dataset Upload → validate & upload the three "
          "files → Train models.")


if __name__ == "__main__":
    main()
