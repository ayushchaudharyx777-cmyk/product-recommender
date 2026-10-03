"""Streamlit dashboard for the hybrid retrieve-and-rank recommender.  Run:  streamlit run app.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
st.set_page_config(page_title="Product Recommender", page_icon="🛒", layout="wide")
RANKER, KNN = "Hybrid ranker (LambdaRank)", "Item-item KNN"


@st.cache_resource
def load_model():
    from recommend import Recommender
    return Recommender()


rec = load_model()
M = json.load(open("reports/metrics.json"))
ts, llo, cold = M["time_split"], M["leave_last_out"], M["cold_start"]


def show(df, drop=("source",)):
    out = df.drop(columns=[c for c in drop if c in df.columns]).rename(columns={
        "product_id": "Product ID", "category_code": "Category", "brand": "Brand", "price": "Price ($)",
        "score": "Score", "why": "Why this product", "source": "Source"})
    out[["Category", "Brand"]] = out[["Category", "Brand"]].fillna("unknown")
    out.index = np.arange(1, len(out) + 1)
    st.dataframe(out, width="stretch")


def label(pid):
    row = rec.info_all.loc[pid]
    cat = row["category_code"] if pd.notna(row["category_code"]) else "unknown"
    brand = row["brand"] if pd.notna(row["brand"]) else "unknown"
    return f"{pid} | {cat} | {brand} | ${row['price']:.2f}" + ("" if int(pid) in rec.i_pos else " | NEW")


st.title("🛒 Personalized Product Recommendation")
st.caption("Hybrid retrieve-and-rank recommender on the REES46 electronics event log: collaborative, sequential, "
           "content and popularity retrieval, re-ranked by LightGBM LambdaRank, with a content-based fallback for "
           "new products and new users.")
c = st.columns(5)
c[0].metric("Users in model", f"{len(rec.users):,}")
c[1].metric("Products in model", f"{len(rec.items):,}")
c[2].metric("Catalogue (with fallback)", f"{len(rec.all_items):,}")
gain = 100 * (ts["models"][RANKER]["ndcg@10"] / ts["models"][KNN]["ndcg@10"] - 1)
c[3].metric("NDCG@10, time-based split", f"{ts['models'][RANKER]['ndcg@10']:.3f}", f"{gain:+.0f}% vs item-KNN")
c[4].metric("Recall@10, time-based split", f"{ts['models'][RANKER]['recall@10']:.3f}")

tabs = st.tabs(["Recommend for a user", "Build a session", "Similar products", "Model comparison",
                "User segments", "Cold start"])

with tabs[0]:
    left, right = st.columns([1, 3])
    with left:
        sample = np.random.default_rng(7).choice(rec.users, size=min(300, len(rec.users)), replace=False)
        uid = st.selectbox("Pick a user", sample)
        custom = st.text_input("...or enter any user ID")
        k = st.slider("Number of recommendations", 5, 50, 10)
        if custom.strip().isdigit():
            uid = int(custom)
    with right:
        if int(uid) in rec.u_pos:
            hist = rec.history(uid)
            st.subheader(f"What user {uid} interacted with ({len(hist)} products)")
            show(hist, drop=("source", "why", "score"))
            st.subheader("Recommended next")
        else:
            st.warning("This user is not in the model. Showing popular products; use the session tab to get "
                       "personalised results for a new visitor.")
        show(rec.for_user(uid, k))

with tabs[1]:
    st.write("Pick the products a visitor has just viewed, in order. This works for anyone, including users the "
             "model has never seen. Products marked NEW are outside the collaborative model and use the "
             "content-based fallback.")
    popular = rec.items[np.argsort(-rec.ret.pop)][:1500]
    new = np.random.default_rng(3).choice(rec.all_items[~np.isin(rec.all_items, rec.items)],
                                          size=min(500, len(rec.all_items) - len(rec.items)), replace=False)
    picked = st.multiselect("Viewed products (type to search)", list(popular) + list(new), format_func=label,
                            default=list(popular[:2]))
    k2 = st.slider("Number of recommendations", 5, 50, 10, key="k2")
    if picked:
        out = rec.for_session(picked, k2)
        st.info(f"Source: {out['source'].iloc[0]}")
        show(out)

with tabs[2]:
    options = list(rec.items[np.argsort(-rec.ret.pop)][:2000]) + list(new)
    pid = st.selectbox("Pick a product (popular first, NEW products at the end; type to search)", options,
                       format_func=label)
    custom_p = st.text_input("...or enter any product ID from the catalogue")
    if custom_p.strip().isdigit():
        pid = int(custom_p)
    k3 = st.slider("Number of similar products", 5, 50, 10, key="k3")
    try:
        out = rec.similar(pid, k3)
        st.info(f"Source: {out['source'].iloc[0]}")
        show(out)
    except KeyError:
        st.error("That product is not in the catalogue.")

with tabs[3]:
    def table(block):
        cols = [f"{m}@{k}" for k in (5, 10, 20) for m in ("precision", "recall", "ndcg")]
        return pd.DataFrame(block["models"]).T[cols].sort_values("ndcg@10", ascending=False)

    st.subheader(f"Time-based split: test on new products after {ts['test_start']} ({ts['n_test_users']:,} users)")
    st.dataframe(table(ts).style.format("{:.4f}").highlight_max(axis=0), width="stretch")
    def ci_line(block):
        ci = block["ranker_vs_knn_ndcg@10"]
        verdict = "significant" if ci["significant"] else "not significant, the interval includes 0"
        return (f"Ranker minus item-KNN on NDCG@10: **{ci['diff']:+.4f}** (95% bootstrap CI {ci['ci_low']:+.4f} to "
                f"{ci['ci_high']:+.4f}; {verdict}). Stage 1 retrieves {block['candidates_per_user']:.0f} candidates "
                f"per user with recall {block['candidate_recall']:.3f}, the ceiling for the ranker.")

    st.write(ci_line(ts))
    st.caption("Only users who come back after the cut-off can be tested, so this test set is small and noisy.")
    st.image("reports/model_comparison.png")
    st.subheader(f"Leave-last-out: predict each user's next product ({llo['n_test_users']:,} users)")
    st.dataframe(table(llo).style.format("{:.4f}").highlight_max(axis=0), width="stretch")
    st.write(ci_line(llo))
    a, b = st.columns(2)
    with a:
        st.subheader("Beyond accuracy (top 10)")
        st.dataframe(pd.DataFrame(llo["beyond_accuracy"]).T, width="stretch")
        st.caption("Coverage: share of the catalogue ever recommended. Diversity: share of pairs in a list from "
                   "different categories. Novelty: higher means less popular products.")
    with b:
        st.subheader("Ablation: ranker without one feature group")
        st.dataframe(pd.DataFrame(llo["ablation"]).T.style.format("{:.4f}"), width="stretch")
    st.image("reports/features_ablation.png")

with tabs[4]:
    seg = pd.read_csv("reports/segments.csv")
    st.write(f"Hybrid ranker on the leave-last-out split ({llo['n_test_users']:,} users), broken down by user "
             "segment. Lift is NDCG@10 against item-KNN.")
    kind = st.radio("Segment by", seg["segment_type"].unique(), horizontal=True)
    s = seg[seg.segment_type == kind].drop(columns="segment_type").set_index("segment")
    st.dataframe(s.style.format({"precision@10": "{:.4f}", "recall@10": "{:.4f}", "ndcg@10": "{:.4f}",
                                 "knn_ndcg@10": "{:.4f}", "lift_vs_knn_pct": "{:+.1f}%"}), width="stretch")
    st.bar_chart(s[["knn_ndcg@10", "ndcg@10"]].rename(columns={"knn_ndcg@10": KNN, "ndcg@10": RANKER}),
                 stack=False)
    st.caption("Small segments are noisy; read the user counts before comparing.")

with tabs[5]:
    st.subheader("Content-based fallback for new users and new products")
    st.write(f"Only {len(rec.users):,} users and {len(rec.items):,} products have enough interactions for "
             f"collaborative filtering. The fallback uses category, brand and price only, so it covers all "
             f"{cold['catalogue_size']:,} catalogue products.")
    st.write(f"Test: {cold['n_users']:,} users outside the model; their last product is hidden and predicted from "
             f"the earlier ones, ranked against the whole catalogue. {cold['targets_outside_cf_model_pct']}% of the "
             f"hidden products are not in the collaborative model at all.")
    st.dataframe(pd.DataFrame(cold["models"]).T.style.format("{:.4f}"), width="stretch")
