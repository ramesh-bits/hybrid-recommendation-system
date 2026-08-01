"""
Streamlit Dashboard – Hybrid Context-Aware Recommender System
Dissertation: M.Tech. AI & ML  |  BITS Pilani

Pages
─────
  1. Recommendations   – top-K with explanation cards
  2. Model Comparison  – Precision@K / Recall@K / NDCG@K charts
  3. Explainability    – detailed model + feature breakdown
  4. Real-Time Sim     – simulate a new rating → watch recs update
  5. Data Explorer     – user/item/rating statistics
  6. Dataset Upload    – load a new dataset (ratings + items CSV)
"""
import os
import json
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(
    page_title="Hybrid Recommender System",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dataset config (drives all labels) ───────────────────────────────────────
@st.cache_data(ttl=60)
def get_dataset_config():
    try:
        r = requests.get(f"{API_URL}/dataset/info", timeout=10)
        return r.json() if r.ok else {}
    except Exception:
        return {}


# ── API helpers ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_evaluation():
    try:
        r = requests.get(f"{API_URL}/evaluation", timeout=10)
        return r.json() if r.ok else {}
    except Exception:
        return {}


def get_recommendations(user_idx, n=10, model="hybrid", hour=12, dow=2, mon=5, **behavior):
    """behavior: optional session context — recent_item, recent_rating,
    recent_view_time, recent_clicked/liked/wishlist/cart/ordered."""
    try:
        params = dict(n=n, model=model, hour=hour, dow=dow, mon=mon)
        if behavior.get("recent_item") is not None:
            params["recent_item"] = behavior["recent_item"]
            params.update({k: v for k, v in behavior.items()
                           if k != "recent_item" and v not in (None, False, 0, 0.0)})
        r = requests.get(f"{API_URL}/recommendations/{user_idx}", params=params, timeout=30)
        return r.json() if r.ok else {"recommendations": []}
    except Exception:
        return {"recommendations": []}


def get_explanation(user_idx, item_idx, hour=12, dow=2, mon=5):
    try:
        params = dict(hour=hour, dow=dow, mon=mon)
        r = requests.get(f"{API_URL}/explain/{user_idx}/{item_idx}", params=params, timeout=20)
        return r.json() if r.ok else {}
    except Exception:
        return {}


def get_history(user_idx, limit=20):
    try:
        r = requests.get(f"{API_URL}/users/{user_idx}/history?limit={limit}", timeout=10)
        return r.json().get("history", []) if r.ok else []
    except Exception:
        return []


def simulate_rating(user_idx, item_idx, rating=None, **events):
    """Submit a rating and/or behavioural events (view_time_sec, clicked,
    liked, wishlist, add_to_cart, ordered) for real-time adaptation."""
    try:
        payload = dict(user_idx=user_idx, item_idx=item_idx, rating=rating, **events)
        r = requests.post(f"{API_URL}/simulate", json=payload, timeout=30)
        return r.json() if r.ok else {}
    except Exception:
        return {}


def health_check():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json() if r.ok else {}
    except Exception:
        return {}


# ── Resolved labels ───────────────────────────────────────────────────────────
cfg           = get_dataset_config()
ITEM_LABEL    = cfg.get("item_label", "item")          # "movie", "product", …
ITEM_LABELS   = ITEM_LABEL + "s"                       # plural
FEAT_LABEL    = cfg.get("feature_label", "Category")  # "Genre", "Category", …
FEAT_LABELS   = FEAT_LABEL + "s"
DATASET_NAME  = cfg.get("dataset_name", "—")
N_USERS       = cfg.get("n_users", 943)
N_ITEMS       = cfg.get("n_items", 1682)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔎 Hybrid Recommender")
    st.caption("M.Tech. Dissertation · BITS Pilani")
    st.divider()

    h = health_check()
    ms = h.get("model_server", {})
    if ms.get("models_loaded"):
        st.success("Models loaded ✓")
    elif ms.get("training"):
        st.warning("Training in progress…")
    else:
        st.error("Models not loaded")
        if st.button("▶ Train models now"):
            requests.post(f"{API_URL}/train", timeout=5)
            st.info("Training started – this takes ~5 min. Refresh in a moment.")

    st.divider()
    page = st.radio(
        "Navigate",
        ["Recommendations", "Model Comparison", "Explainability",
         "Real-Time Simulation", "Data Explorer", "Dataset Upload"],
    )

    st.divider()
    user_idx = st.number_input("User index", min_value=0, max_value=max(N_USERS - 1, 0), value=0, step=1)
    n_recs   = st.slider(f"# Recommendations (K)", 5, 30, 10)

    st.divider()
    st.subheader("Temporal context")
    hour = st.slider("Hour of day", 0, 23, int(datetime.now().hour))
    dow  = st.slider("Day of week (Mon=0)", 0, 6, datetime.now().weekday())
    mon  = st.slider("Month (Jan=0)", 0, 11, datetime.now().month - 1)

    st.subheader("Behavioural context")
    st.caption(
        "Optional session signal: what the user just did. Applied to the next "
        "recommendation request only — nothing is stored."
    )
    use_behavior = st.checkbox("Add a recent interaction")
    behavior_ctx = {}
    if use_behavior:
        b_item = st.number_input(f"Recently interacted {ITEM_LABEL} (item index)",
                                 0, max(N_ITEMS - 1, 0), 0)
        b_view = st.slider("View time (seconds)", 0, 600, 0, 10)
        b_click = st.checkbox("Clicked / opened", key="bc_click")
        b_like  = st.checkbox("Liked 👍", key="bc_like")
        b_wish  = st.checkbox("Wishlist ⭐", key="bc_wish")
        b_cart  = st.checkbox("Add to cart 🛒", key="bc_cart")
        b_order = st.checkbox("Ordered ✅", key="bc_order")
        behavior_ctx = dict(
            recent_item=b_item, recent_view_time=float(b_view),
            recent_clicked=b_click, recent_liked=b_like, recent_wishlist=b_wish,
            recent_cart=b_cart, recent_ordered=b_order,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 – Recommendations
# ─────────────────────────────────────────────────────────────────────────────
if page == "Recommendations":
    st.header(f"🎯 Personalised Recommendations")

    model_choice = st.selectbox("Algorithm", ["hybrid", "cf", "cbf", "ncf"])

    col_load, _ = st.columns([1, 3])
    with col_load:
        fetch = st.button("Get Recommendations", type="primary")

    if fetch or "recs_data" not in st.session_state:
        with st.spinner("Fetching…"):
            st.session_state["recs_data"] = get_recommendations(
                user_idx, n=n_recs, model=model_choice, hour=hour, dow=dow, mon=mon,
                **behavior_ctx,
            )

    data = st.session_state["recs_data"]
    recs = data.get("recommendations", [])

    bc = data.get("behavior_context")
    if bc and bc.get("applied"):
        st.info(
            f"🧭 Behavioural session context applied: recent interaction with item "
            f"{bc['item_idx']} (engagement weight {bc['engagement_weight']}, "
            f"effective rating {bc['effective_rating']}). Rankings below reflect it; "
            "nothing was stored."
        )

    if not recs:
        st.warning(f"No recommendations returned. Check that models are trained.")
    else:
        st.subheader(f"Top {len(recs)} for user {user_idx}  ·  model: {model_choice}")
        for r in recs:
            with st.expander(f"#{r['rank']}  {r['name']}   —  score {r['score']:.3f}"):
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.write(f"**{FEAT_LABELS}:**", ", ".join(r.get("features", [])) or "—")
                    exp = get_explanation(user_idx, r["item_idx"], hour, dow, mon)
                    if exp.get("natural_language"):
                        st.info(f"💡 {exp['natural_language']}")
                    if exp.get("feature_match"):
                        fm = exp["feature_match"]
                        fig = go.Figure(go.Bar(
                            x=list(fm.values()), y=list(fm.keys()),
                            orientation="h", marker_color="steelblue",
                        ))
                        fig.update_layout(
                            title=f"{FEAT_LABEL} match with your history",
                            xaxis_title=f"Fraction of liked {ITEM_LABELS}",
                            height=max(150, 30 * len(fm)),
                            margin=dict(l=0, r=0, t=30, b=0),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                with c2:
                    mc = exp.get("model_contributions", {})
                    contrib = {
                        k: v["contribution"]
                        for k, v in mc.items()
                        if k != "hybrid_score" and isinstance(v, dict)
                    }
                    if contrib:
                        fig2 = go.Figure(go.Pie(
                            labels=list(contrib.keys()),
                            values=list(contrib.values()),
                            hole=0.4,
                        ))
                        fig2.update_layout(
                            title="Model contribution",
                            height=250,
                            margin=dict(l=0, r=0, t=40, b=0),
                            showlegend=True,
                        )
                        st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 – Model Comparison
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Model Comparison":
    st.header("📊 Model Performance Comparison")
    st.caption("Metrics evaluated on a time-based held-out test set.")

    eval_data = get_evaluation()
    if not eval_data:
        st.warning("No evaluation data. Train models first.")
    else:
        meta        = eval_data.pop("_meta", {})   # run info, not a model
        models      = [m for m in eval_data if not m.startswith("_")]
        metric_keys = list(next(iter(eval_data.values())).keys())

        if meta.get("hybrid_weights"):
            w = meta["hybrid_weights"]
            st.info(
                f"Tuned hybrid weights — CF: {w.get('cf')}, CBF: {w.get('cbf')}, "
                f"NCF: {w.get('ncf')}  ·  evaluated users: {meta.get('eval_users', '—')}"
            )

        k_vals = sorted(set(int(m.split("@")[1]) for m in metric_keys if "@" in m))
        k_sel  = st.selectbox("K", k_vals, index=0)

        for metric_prefix in ["precision", "recall", "ndcg", "mrr"]:
            key = f"{metric_prefix}@{k_sel}"
            if key not in metric_keys:
                continue
            values = [eval_data[m].get(key, 0) for m in models]

            fig = go.Figure(go.Bar(
                x=models,
                y=values,
                marker_color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"],
                text=[f"{v:.4f}" for v in values],
                textposition="outside",
            ))
            fig.update_layout(
                title=f"{key.upper()}",
                yaxis_title="Score",
                yaxis_range=[0, min(1.0, max(values) * 1.3)],
                height=320,
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Full metric table")
        rows = []
        for m in models:
            row = {"Model": m.upper()}
            row.update(eval_data[m])
            rows.append(row)
        df = pd.DataFrame(rows).set_index("Model")
        st.dataframe(df.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 – Explainability
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Explainability":
    st.header("🔍 Recommendation Explainability")

    item_idx = st.number_input("Item index to explain", min_value=0, max_value=max(N_ITEMS - 1, 0), value=50)

    if st.button("Explain this recommendation", type="primary"):
        with st.spinner("Generating explanation…"):
            exp = get_explanation(user_idx, item_idx, hour, dow, mon)

        if not exp:
            st.error("Could not generate explanation.")
        else:
            st.subheader(f"Why recommend '{exp.get('name', item_idx)}' to user {user_idx}?")
            st.info(f"**Natural language:** {exp.get('natural_language', '—')}")

            c1, c2 = st.columns(2)

            mc      = exp.get("model_contributions", {})
            contrib = {k: v["contribution"] for k, v in mc.items()
                       if k != "hybrid_score" and isinstance(v, dict)}
            if contrib:
                with c1:
                    st.subheader("Model contributions")
                    fig = go.Figure(go.Pie(
                        labels=[k.replace("_", " ").title() for k in contrib],
                        values=list(contrib.values()),
                        hole=0.45,
                        marker_colors=["#4C72B0", "#55A868", "#DD8452"],
                    ))
                    fig.update_layout(height=320, margin=dict(t=20))
                    st.plotly_chart(fig, use_container_width=True)

                    rows = []
                    for k, v in mc.items():
                        if k == "hybrid_score":
                            continue
                        rows.append({
                            "Model":          k.replace("_", " ").title(),
                            "Raw score":      round(v["raw_score"], 4),
                            "Weight":         v["weight"],
                            "Contribution %": f"{v['contribution']*100:.1f}%",
                        })
                    st.table(pd.DataFrame(rows))

            fm = exp.get("feature_match", {})
            if fm:
                with c2:
                    st.subheader(f"{FEAT_LABEL} alignment with your history")
                    fig2 = go.Figure(go.Bar(
                        x=list(fm.values()),
                        y=list(fm.keys()),
                        orientation="h",
                        marker_color="steelblue",
                        text=[f"{v*100:.1f}%" for v in fm.values()],
                        textposition="auto",
                    ))
                    fig2.update_layout(
                        xaxis_title=f"% of your liked {ITEM_LABELS} in this {FEAT_LABEL.lower()}",
                        height=320,
                        margin=dict(t=20),
                    )
                    st.plotly_chart(fig2, use_container_width=True)

            sl = exp.get("similar_liked", [])
            if sl:
                st.subheader(f"{ITEM_LABEL.capitalize()}s you liked that share {FEAT_LABELS.lower()}")
                for item in sl:
                    st.write(f"• **{item['name']}**  ({item['genre_overlap']} {FEAT_LABELS.lower()} in common)")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 – Real-Time Simulation
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Real-Time Simulation":
    st.header("⚡ Real-Time Adaptation Simulation")
    st.caption(
        f"Simulate an interaction with a {ITEM_LABEL} — an explicit rating and/or "
        "behavioural events such as view time, clicks, wishlist, add-to-cart, and "
        "orders — and watch the system instantly update the user profile and "
        "re-rank recommendations. Stronger engagement produces a stronger update."
    )

    c1, c2 = st.columns(2)
    with c1:
        new_item_idx = st.number_input("Item index to interact with", 0, max(N_ITEMS - 1, 0), 100)

        give_rating = st.checkbox("Give an explicit rating", value=True)
        new_rating  = st.slider("Your rating", 1.0, 5.0, 4.0, 0.5,
                                disabled=not give_rating)

        st.markdown("**Behavioural signals** *(optional — engagement raises the update strength)*")
        view_time = st.slider("Time spent viewing (seconds)", 0, 600, 0, 10)
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            ev_clicked  = st.checkbox("Clicked / opened")
            ev_liked    = st.checkbox("Liked 👍")
        with bc2:
            ev_wishlist = st.checkbox("Wishlist ⭐")
            ev_cart     = st.checkbox("Add to cart 🛒")
        with bc3:
            ev_ordered  = st.checkbox("Ordered ✅")

    with c2:
        st.write(f"**Before adaptation** – current top-5")
        before = get_recommendations(user_idx, n=5, model="hybrid", hour=hour, dow=dow, mon=mon)
        for r in before.get("recommendations", []):
            st.write(f"  {r['rank']}. {r['name']}")

    any_event = any([view_time > 0, ev_clicked, ev_liked, ev_wishlist, ev_cart, ev_ordered])
    if st.button("Submit interaction & adapt", type="primary",
                 disabled=not (give_rating or any_event)):
        with st.spinner("Updating profile and re-ranking…"):
            result = simulate_rating(
                user_idx, new_item_idx,
                rating=new_rating if give_rating else None,
                view_time_sec=float(view_time),
                clicked=ev_clicked, liked=ev_liked, wishlist=ev_wishlist,
                add_to_cart=ev_cart, ordered=ev_ordered,
            )

        st.success("Profile updated!")
        m1, m2, m3 = st.columns(3)
        m1.metric("Engagement weight", result.get("engagement_weight", "—"),
                  help="1.0 = plain rating; behavioural signals add up to ~4.0")
        m2.metric("Effective rating", result.get("effective_rating", "—"),
                  help="Explicit rating, or inferred from engagement if none given")
        m3.metric("Profile update strength (α)", result.get("profile_update_alpha", "—"),
                  help="EMA step size applied to the user profile")
        if result.get("signals_applied"):
            st.caption("Signals applied: " + ", ".join(result["signals_applied"]))
        after_recs = result.get("updated_recommendations", [])

        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Before")
            for r in before.get("recommendations", []):
                st.write(f"  {r['rank']}. {r['name']}")
        with c4:
            st.subheader("After (real-time update)")
            for r in after_recs[:5]:
                st.write(f"  {r['rank']}. {r['name']}")

        before_names = {r["name"] for r in before.get("recommendations", [])[:5]}
        after_names  = {r["name"] for r in after_recs[:5]}
        new_entries  = after_names - before_names
        if new_entries:
            st.info(f"New entries after adaptation: {', '.join(new_entries)}")
        else:
            st.info(f"Top-5 unchanged – try rating with a very different score.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 – Data Explorer
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Data Explorer":
    st.header(f"🗂️ Dataset Explorer  –  {DATASET_NAME}")

    tab1, tab2 = st.tabs(["User history", "Dataset overview"])

    with tab1:
        hist = get_history(user_idx, limit=50)
        if not hist:
            st.info("No history found for this user index.")
        else:
            df = pd.DataFrame(hist)
            st.write(f"**User {user_idx}** has rated {len(df)} {ITEM_LABELS}.")

            col_a, col_b = st.columns(2)
            with col_a:
                fig = px.histogram(df, x="rating", nbins=9, title="Rating distribution",
                                   color_discrete_sequence=["steelblue"])
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                if "features" in df.columns:
                    feat_counts: dict = {}
                    for feats in df["features"]:
                        for f in feats:
                            feat_counts[f] = feat_counts.get(f, 0) + 1
                    fdf = pd.DataFrame(feat_counts.items(), columns=[FEAT_LABEL, "Count"])
                    fdf = fdf.sort_values("Count", ascending=False)
                    fig2 = px.bar(fdf, x="Count", y=FEAT_LABEL, orientation="h",
                                  title=f"{FEAT_LABELS} encountered",
                                  color_discrete_sequence=["#4C72B0"])
                    st.plotly_chart(fig2, use_container_width=True)

            st.dataframe(
                df[["name", "features", "rating"]].rename(
                    columns={"name": "Name", "features": FEAT_LABELS, "rating": "Rating"}
                ),
                use_container_width=True,
            )

    with tab2:
        st.markdown(f"""
        | Attribute | Value |
        |-----------|-------|
        | Dataset   | {DATASET_NAME} |
        | Users     | {N_USERS} |
        | Items     | {N_ITEMS} |
        | Time span | Sep 1997 – Apr 1998 *(MovieLens default)* |
        """)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 – Dataset Upload
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Dataset Upload":
    st.header("📤 Load a New Dataset")
    st.markdown(
        "Upload your own dataset to replace the current one. Review the expected "
        "schema below, **validate** your files first, then upload. After uploading, "
        "click **Train models** to build the recommender on the new data."
    )

    # ── Expected schema (served by the API — single source of truth) ─────
    @st.cache_data(ttl=300)
    def get_schema():
        try:
            r = requests.get(f"{API_URL}/schema", timeout=10)
            return r.json() if r.ok else {}
        except Exception:
            return {}

    schema_data = get_schema()
    schema      = schema_data.get("schema", {})
    templates   = schema_data.get("templates", {})

    st.subheader("Expected schema")
    if schema:
        tab_r, tab_i, tab_c = st.tabs(["ratings.csv", "items.csv", "config.json"])
        with tab_r:
            st.markdown("**Required columns**")
            st.table(pd.DataFrame(schema["ratings"]["required"]))
            st.markdown("**Optional behavioural columns** *(raise the confidence "
                        "weight of an interaction during training)*")
            st.table(pd.DataFrame(schema["ratings"]["optional"]))
        with tab_i:
            st.markdown("**Required columns**")
            st.table(pd.DataFrame(schema["items"]["required"]))
        with tab_c:
            st.table(pd.DataFrame(schema["config"]))

        if templates:
            st.markdown("**Download sample templates:**")
            t1, t2, t3 = st.columns(3)
            t1.download_button("⬇ ratings.csv template", templates.get("ratings_csv", ""),
                               "ratings_template.csv", "text/csv")
            t2.download_button("⬇ items.csv template", templates.get("items_csv", ""),
                               "items_template.csv", "text/csv")
            t3.download_button("⬇ config.json template", templates.get("config_json", ""),
                               "config_template.json", "application/json")
    else:
        st.warning("Could not load the schema from the API — is the backend running?")

    st.divider()
    st.subheader("🧪 Load a demo dataset (one click)")
    st.caption(
        "Downloads a real public dataset — Amazon product reviews "
        "(UCSD/McAuley, 2018), including the verified-purchase flag as the "
        "'ordered' behavioural signal — converts it to the schema above, "
        "validates it, and loads it. No files needed."
    )
    d1, d2, d3 = st.columns([2, 2, 1])
    with d1:
        demo_cat = st.selectbox(
            "Category",
            ["Software", "Luxury_Beauty", "Musical_Instruments"],
            help="Software ≈ 13K ratings (fastest); Musical_Instruments ≈ 230K",
        )
    with d2:
        demo_cap = st.number_input(
            "Max ratings (0 = all)", 0, 500000, 0, 10000,
            help="Keeps the most recent N ratings to bound training time",
        )
    with d3:
        st.write("")
        demo_go = st.button("⬇ Load demo", type="primary")
        demo_prep = st.button("📄 Get CSVs only",
                              help="Prepare the same dataset but download the three "
                                   "files instead of loading them — for demonstrating "
                                   "the manual upload flow")

    if demo_prep:
        with st.spinner(f"Preparing '{demo_cat}' files…"):
            try:
                resp = requests.post(
                    f"{API_URL}/demo-dataset/prepare",
                    json={"category": demo_cat,
                          "max_ratings": int(demo_cap) or None},
                    timeout=600,
                )
                if resp.ok:
                    st.session_state["demo_files"] = resp.json()
                else:
                    st.error(f"Prepare failed ({resp.status_code}): {resp.text}")
            except Exception as e:
                st.error(f"Request error: {e}")

    if st.session_state.get("demo_files"):
        df_ = st.session_state["demo_files"]
        st.success(
            f"'{df_['dataset_name']}' prepared: {df_['n_ratings']:,} ratings, "
            f"{df_['n_users']:,} users, {df_['n_items']:,} products. "
            "Download the files, then upload them in the section below."
        )
        p1, p2, p3 = st.columns(3)
        p1.download_button("⬇ ratings.csv", df_["ratings_csv"], "ratings.csv", "text/csv")
        p2.download_button("⬇ items.csv", df_["items_csv"], "items.csv", "text/csv")
        p3.download_button("⬇ config.json", df_["config_json"], "config.json", "application/json")

    if demo_go:
        with st.spinner(f"Downloading and ingesting '{demo_cat}' — this can take a few minutes…"):
            try:
                resp = requests.post(
                    f"{API_URL}/demo-dataset",
                    json={"category": demo_cat,
                          "max_ratings": int(demo_cap) or None},
                    timeout=600,
                )
                if resp.ok:
                    result = resp.json()
                    rep = result.get("validation", {})
                    for w in rep.get("warnings", []):
                        st.warning(f"⚠️ {w}")
                    st.success(
                        f"Dataset '{result['dataset_name']}' loaded: "
                        f"{result['n_users']:,} users, {result['n_items']:,} products, "
                        f"categories: {', '.join(result['feature_cols'][:6])}…"
                    )
                    st.info("Now click '▶ Train models now' in the sidebar to train on it (~2-5 min).")
                    get_dataset_config.clear()
                else:
                    st.error(f"Demo load failed ({resp.status_code}): {resp.text}")
            except Exception as e:
                st.error(f"Request error: {e}")

    st.divider()
    st.subheader("Current dataset")
    info_cols = st.columns(4)
    info_cols[0].metric("Dataset", DATASET_NAME)
    info_cols[1].metric("Users", N_USERS)
    info_cols[2].metric("Items", N_ITEMS)
    info_cols[3].metric(f"{FEAT_LABEL}s", len(cfg.get("feature_cols", [])))

    st.divider()
    st.subheader("Upload new dataset")

    ratings_file = st.file_uploader("ratings.csv", type="csv")
    items_file   = st.file_uploader("items.csv",   type="csv")
    config_text  = st.text_area(
        "DatasetConfig (JSON)",
        value=json.dumps({
            "dataset_name":     "my-dataset",
            "item_label":       "item",
            "feature_cols":     ["Category_A", "Category_B"],
            "feature_label":    "Category",
            "rating_threshold": 3.5,
            "item_name_col":    "name",
        }, indent=2),
        height=220,
    )

    def _post_files(path):
        return requests.post(
            f"{API_URL}{path}",
            files={
                "ratings_file": (ratings_file.name, ratings_file.getvalue(), "text/csv"),
                "items_file":   (items_file.name,   items_file.getvalue(),   "text/csv"),
            },
            data={"config_json": config_text},
            timeout=120,
        )

    def _files_ready():
        if not ratings_file or not items_file:
            st.error("Please upload both ratings.csv and items.csv.")
            return False
        try:
            json.loads(config_text)
        except json.JSONDecodeError as e:
            st.error(f"Config is not valid JSON: {e}")
            return False
        return True

    def _show_report(report):
        for e in report.get("errors", []):
            st.error(f"❌ {e}")
        for w in report.get("warnings", []):
            st.warning(f"⚠️ {w}")
        stats = report.get("stats", {})
        if stats:
            c = st.columns(4)
            c[0].metric("Ratings", f"{stats.get('n_ratings', 0):,}")
            c[1].metric("Users", f"{stats.get('n_users', 0):,}")
            c[2].metric("Items", f"{stats.get('n_items', 0):,}")
            c[3].metric("Rating range",
                        f"{stats.get('rating_min', '—')} – {stats.get('rating_max', '—')}")
            beh = stats.get("behavioural_columns_detected", [])
            st.caption("Behavioural columns detected: "
                       + (", ".join(beh) if beh else "none (all interactions weighted equally)"))

    b1, b2 = st.columns(2)
    with b1:
        validate_clicked = st.button("🔍 Validate files", help="Dry run — checks the files, ingests nothing")
    with b2:
        upload_clicked = st.button("📤 Validate & Upload", type="primary")

    if validate_clicked and _files_ready():
        with st.spinner("Validating…"):
            try:
                resp = _post_files("/validate")
                if resp.ok:
                    report = resp.json()
                    _show_report(report)
                    if report.get("valid"):
                        st.success("✅ Files are valid — ready to upload.")
                    else:
                        st.error("Fix the errors above, then validate again.")
                else:
                    st.error(f"Validation request failed ({resp.status_code}): {resp.text}")
            except Exception as e:
                st.error(f"Request error: {e}")

    if upload_clicked and _files_ready():
        with st.spinner("Validating and ingesting dataset…"):
            try:
                v = _post_files("/validate")
                report = v.json() if v.ok else {}
                if v.ok and not report.get("valid", False):
                    _show_report(report)
                    st.error("Upload blocked — fix the errors above first.")
                else:
                    resp = _post_files("/upload")
                    if resp.ok:
                        result = resp.json()
                        _show_report(report)
                        st.success(
                            f"Dataset '{result['dataset_name']}' loaded: "
                            f"{result['n_users']} users, {result['n_items']} items."
                        )
                        get_dataset_config.clear()
                        st.session_state["dataset_ready_to_train"] = result["dataset_name"]
                    else:
                        st.error(f"Upload failed ({resp.status_code}): {resp.text}")
            except Exception as e:
                st.error(f"Request error: {e}")

    # Rendered OUTSIDE the upload-click branch so the button survives reruns
    if st.session_state.get("dataset_ready_to_train"):
        st.divider()
        st.subheader("Next step: train")
        st.write(
            f"Dataset **{st.session_state['dataset_ready_to_train']}** is loaded "
            "but models are not trained yet."
        )
        if st.button("▶ Train models on the new dataset", type="primary"):
            try:
                requests.post(f"{API_URL}/train", timeout=10)
                st.session_state.pop("dataset_ready_to_train", None)
                st.info(
                    "Training started (~2-5 min depending on dataset size). "
                    "The sidebar status will show 'Models loaded ✓' when done — "
                    "then open the Recommendations page."
                )
            except Exception as e:
                st.error(f"Could not start training: {e}")
