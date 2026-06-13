"""
Streamlit Dashboard – Hybrid Context-Aware Recommender System
Dissertation: M.Tech. AI & ML  |  BITS Pilani

Pages
─────
  1. Recommendations   – top-K with explanation cards
  2. Model Comparison  – Precision@K / Recall@K / NDCG@K charts
  3. Explainability    – detailed model + genre breakdown
  4. Real-Time Sim     – simulate a new rating → watch recs update
  5. Data Explorer     – user/item/rating statistics
"""
import os
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
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_evaluation():
    try:
        r = requests.get(f"{API_URL}/evaluation", timeout=10)
        return r.json() if r.ok else {}
    except Exception:
        return {}


def get_recommendations(user_idx, n=10, model="hybrid", hour=12, dow=2, mon=5):
    try:
        params = dict(n=n, model=model, hour=hour, dow=dow, mon=mon)
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


def simulate_rating(user_idx, item_idx, rating):
    try:
        r = requests.post(f"{API_URL}/simulate",
                          json=dict(user_idx=user_idx, item_idx=item_idx, rating=rating),
                          timeout=30)
        return r.json() if r.ok else {}
    except Exception:
        return {}


def health_check():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json() if r.ok else {}
    except Exception:
        return {}


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎬 Hybrid Recommender")
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
         "Real-Time Simulation", "Data Explorer"],
    )

    st.divider()
    user_idx = st.number_input("User index", min_value=0, max_value=942, value=0, step=1)
    n_recs   = st.slider("# Recommendations (K)", 5, 30, 10)

    st.divider()
    st.subheader("Context")
    hour = st.slider("Hour of day", 0, 23, int(datetime.now().hour))
    dow  = st.slider("Day of week (Mon=0)", 0, 6, datetime.now().weekday())
    mon  = st.slider("Month (Jan=0)", 0, 11, datetime.now().month - 1)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 – Recommendations
# ─────────────────────────────────────────────────────────────────────────────
if page == "Recommendations":
    st.header("🎯 Personalised Recommendations")

    model_choice = st.selectbox("Algorithm", ["hybrid", "cf", "cbf", "ncf"])

    col_load, _ = st.columns([1, 3])
    with col_load:
        fetch = st.button("Get Recommendations", type="primary")

    if fetch or "recs_data" not in st.session_state:
        with st.spinner("Fetching…"):
            st.session_state["recs_data"] = get_recommendations(
                user_idx, n=n_recs, model=model_choice, hour=hour, dow=dow, mon=mon
            )

    data  = st.session_state["recs_data"]
    recs  = data.get("recommendations", [])

    if not recs:
        st.warning("No recommendations returned. Check that models are trained.")
    else:
        st.subheader(f"Top {len(recs)} for user {user_idx}  ·  model: {model_choice}")
        for r in recs:
            with st.expander(f"#{r['rank']}  {r['title']}   —  score {r['score']:.3f}"):
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.write("**Genres:**", ", ".join(r["genres"]) or "—")
                    # Inline explanation
                    exp = get_explanation(user_idx, r["item_idx"], hour, dow, mon)
                    if exp.get("natural_language"):
                        st.info(f"💡 {exp['natural_language']}")
                    if exp.get("genre_match"):
                        gm = exp["genre_match"]
                        fig = go.Figure(go.Bar(
                            x=list(gm.values()), y=list(gm.keys()),
                            orientation="h",
                            marker_color="steelblue",
                        ))
                        fig.update_layout(
                            title="Genre match with your history",
                            xaxis_title="Fraction of liked films",
                            height=max(150, 30 * len(gm)),
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
        models = list(eval_data.keys())
        metric_keys = list(next(iter(eval_data.values())).keys())

        # Parse available K values
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

        # Table view
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

    item_idx = st.number_input("Item index to explain", min_value=0, max_value=1681, value=50)

    if st.button("Explain this recommendation", type="primary"):
        with st.spinner("Generating explanation…"):
            exp = get_explanation(user_idx, item_idx, hour, dow, mon)

        if not exp:
            st.error("Could not generate explanation.")
        else:
            st.subheader(f"Why recommend '{exp.get('title', item_idx)}' to user {user_idx}?")

            st.info(f"**Natural language:** {exp.get('natural_language', '—')}")

            c1, c2 = st.columns(2)

            # Model contributions
            mc = exp.get("model_contributions", {})
            contrib = {
                k: v["contribution"]
                for k, v in mc.items()
                if k != "hybrid_score" and isinstance(v, dict)
            }
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
                            "Model": k.replace("_", " ").title(),
                            "Raw score": round(v["raw_score"], 4),
                            "Weight": v["weight"],
                            "Contribution %": f"{v['contribution']*100:.1f}%",
                        })
                    st.table(pd.DataFrame(rows))

            # Genre match
            gm = exp.get("genre_match", {})
            if gm:
                with c2:
                    st.subheader("Genre alignment with your history")
                    fig2 = go.Figure(go.Bar(
                        x=list(gm.values()),
                        y=[g.replace("Children's", "Children") for g in gm.keys()],
                        orientation="h",
                        marker_color="steelblue",
                        text=[f"{v*100:.1f}%" for v in gm.values()],
                        textposition="auto",
                    ))
                    fig2.update_layout(
                        xaxis_title="% of your liked films in this genre",
                        height=320,
                        margin=dict(t=20),
                    )
                    st.plotly_chart(fig2, use_container_width=True)

            # Similar liked items
            sl = exp.get("similar_liked", [])
            if sl:
                st.subheader("Films you liked that share genres")
                for item in sl:
                    st.write(f"• **{item['title']}**  ({item['genre_overlap']} genres in common)")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 – Real-Time Simulation
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Real-Time Simulation":
    st.header("⚡ Real-Time Adaptation Simulation")
    st.caption(
        "Rate a new movie and watch how the system instantly updates "
        "the content-based user profile and re-ranks recommendations."
    )

    c1, c2 = st.columns(2)
    with c1:
        new_item_idx = st.number_input("Item index to rate", 0, 1681, 100)
        new_rating   = st.slider("Your rating", 1.0, 5.0, 4.0, 0.5)

    with c2:
        st.write("**Before adaptation** – current top-5")
        before = get_recommendations(user_idx, n=5, model="hybrid", hour=hour, dow=dow, mon=mon)
        for r in before.get("recommendations", []):
            st.write(f"  {r['rank']}. {r['title']}")

    if st.button("Submit rating & adapt", type="primary"):
        with st.spinner("Updating profile and re-ranking…"):
            result = simulate_rating(user_idx, new_item_idx, new_rating)

        st.success("Profile updated!")
        after_recs = result.get("updated_recommendations", [])

        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Before")
            for r in before.get("recommendations", []):
                st.write(f"  {r['rank']}. {r['title']}")
        with c4:
            st.subheader("After (real-time update)")
            for r in after_recs[:5]:
                st.write(f"  {r['rank']}. {r['title']}")

        # Highlight changes
        before_titles = {r["title"] for r in before.get("recommendations", [])[:5]}
        after_titles  = {r["title"] for r in after_recs[:5]}
        new_entries   = after_titles - before_titles
        if new_entries:
            st.info(f"New entries after adaptation: {', '.join(new_entries)}")
        else:
            st.info("Top-5 unchanged – try rating with a very different score.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 – Data Explorer
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Data Explorer":
    st.header("🗂️ Dataset Explorer  –  MovieLens 100K")

    tab1, tab2 = st.tabs(["User history", "Dataset overview"])

    with tab1:
        hist = get_history(user_idx, limit=50)
        if not hist:
            st.info("No history found for this user index.")
        else:
            df = pd.DataFrame(hist)
            st.write(f"**User {user_idx}** has rated {len(df)} movies.")

            col_a, col_b = st.columns(2)
            with col_a:
                fig = px.histogram(df, x="rating", nbins=9, title="Rating distribution",
                                   color_discrete_sequence=["steelblue"])
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                # Genre breakdown
                if "genres" in df.columns:
                    genre_counts: dict = {}
                    for genres in df["genres"]:
                        for g in genres:
                            genre_counts[g] = genre_counts.get(g, 0) + 1
                    gdf = pd.DataFrame(genre_counts.items(), columns=["Genre", "Count"])
                    gdf = gdf.sort_values("Count", ascending=False)
                    fig2 = px.bar(gdf, x="Count", y="Genre", orientation="h",
                                  title="Genres watched", color_discrete_sequence=["#4C72B0"])
                    st.plotly_chart(fig2, use_container_width=True)

            st.dataframe(
                df[["title", "genres", "rating"]].rename(
                    columns={"title": "Title", "genres": "Genres", "rating": "Rating"}
                ),
                use_container_width=True,
            )

    with tab2:
        st.markdown("""
        | Attribute | Value |
        |-----------|-------|
        | Dataset   | MovieLens 100K |
        | Users     | 943 |
        | Items     | 1,682 movies |
        | Ratings   | 100,000 (scale 1–5) |
        | Sparsity  | ~93.7% |
        | Time span | Sep 1997 – Apr 1998 |
        | Source    | [grouplens.org](https://grouplens.org/datasets/movielens/100k/) |
        """)
