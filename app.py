"""Streamlit demo for the hybrid retrieve-and-rank recommender.
Run from project root:  streamlit run app.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
st.set_page_config(page_title="Product Recommender", page_icon="🛒", layout="wide")


@st.cache_resource
def load_model():
    import recommend
    return recommend


r = load_model()
users, items, info = r.a["users"], r.a["items"], r.info


def tidy(df):
    out = df.reset_index().rename(columns={"product_id": "Product ID", "category_code": "Category",
                                           "brand": "Brand", "price": "Price ($)", "score": "Score"})
    out["Category"] = out["Category"].fillna("unknown")
    out["Brand"] = out["Brand"].fillna("unknown")
    out.index = np.arange(1, len(out) + 1)
    return out


st.title("🛒 Personalized Product Recommendation")
st.caption("Hybrid retrieve-and-rank recommender on the REES46 electronics store event log: "
           "item-KNN + sequential + popularity retrieval, re-ranked by LightGBM LambdaRank.")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Users in model", f"{len(users):,}")
c2.metric("Products in model", f"{len(items):,}")
mr = json.load(open("models/metrics_ranker.json"))
mb = json.load(open("models/metrics.json"))
hit, knn_hit = mr["metrics"]["hybrid ranker"]["hit@10"], mr["metrics"]["item_knn only"]["hit@10"]
c3.metric("Hit@10 (hybrid ranker)", f"{hit:.3f}", f"{100 * (hit / knn_hit - 1):+.0f}% vs item-KNN")
c4.metric("Candidate recall", f"{mr['candidate_recall']:.3f}")

tab_user, tab_item, tab_model = st.tabs(["Recommend for a user", "Similar products", "Model performance"])

with tab_user:
    left, right = st.columns([1, 3])
    with left:
        sample = np.random.default_rng(7).choice(users, size=min(300, len(users)), replace=False)
        uid = st.selectbox("Pick a user", sample)
        custom = st.text_input("...or enter any user ID")
        k = st.slider("Number of recommendations", 5, 50, 20)
        if custom.strip().isdigit():
            uid = int(custom)
    with right:
        if uid in r.u_pos:
            hist = info.iloc[r.p.X[r.u_pos[uid]].indices]
            st.subheader(f"What user {uid} interacted with ({len(hist)} products)")
            st.dataframe(tidy(hist), width="stretch")
            st.subheader("Recommended next")
        else:
            st.warning("This user is not in the model (fewer than 3 products). Showing most popular products.")
        st.dataframe(tidy(r.for_user(uid, k)), width="stretch")

with tab_item:
    labels = {int(pid): f"{pid} | {row.category_code if pd.notna(row.category_code) else 'unknown'} | "
                        f"{row.brand if pd.notna(row.brand) else 'unknown'} | ${row.price:.2f}"
              for pid, row in info.iterrows()}
    popular = items[np.argsort(-r.p.pop)]
    pid = st.selectbox("Pick a product (sorted by popularity, type to search)", popular,
                       format_func=lambda x: labels[int(x)])
    k2 = st.slider("Number of similar products", 5, 20, 10, key="k2")
    st.dataframe(tidy(r.similar_items(pid, k2)), width="stretch")

with tab_model:
    rows = {"Popularity (baseline)": mb["metrics"]["popularity"], "SVD (64 factors)": mb["metrics"]["svd"],
            "Item-item KNN": mr["metrics"]["item_knn only"],
            "Sequential transitions": mr["metrics"]["sequence only"],
            "Hybrid ranker (LambdaRank)": mr["metrics"]["hybrid ranker"]}
    res = pd.DataFrame(rows).T[["hit@10", "ndcg@10", "mrr@10"]]
    a, b = st.columns(2)
    with a:
        st.subheader("Leave-last-out evaluation")
        st.dataframe(res.style.format("{:.4f}").highlight_max(axis=0), width="stretch")
        st.bar_chart(res["hit@10"])
    with b:
        st.subheader("Ranker feature importance (gain %)")
        st.bar_chart(pd.Series(mr["feature_importance_pct"]).sort_values(ascending=False))
    st.info(f"Stage 1 retrieves about {mr['candidates_per_user']:.0f} candidates per user with recall "
            f"{mr['candidate_recall']:.3f}. That recall is the upper bound for the ranker.")