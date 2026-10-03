"""Usage:
  python src/recommend.py --user 1515915625519388267
  python src/recommend.py --item 1996170
"""
import argparse

import joblib
import numpy as np

m = joblib.load("models/recommender.joblib")
u_pos = {u: i for i, u in enumerate(m["users"])}
i_pos = {p: i for i, p in enumerate(m["items"])}


def _show(idx, scores):
    out = m["meta"].iloc[idx].copy()
    out["score"] = np.round(scores, 4)
    return out


def for_user(user_id, k=10):
    if user_id not in u_pos:  # cold start -> popular items
        idx = np.argsort(-m["pop"])[:k]
        return _show(idx, m["pop"][idx])
    row = m["X"][u_pos[user_id]]
    sc = (row @ m["S"]).toarray().ravel()
    sc[row.indices] = -np.inf
    idx = np.argsort(-sc)[:k]
    return _show(idx, sc[idx])


def similar_items(product_id, k=10):
    if product_id not in i_pos:
        raise SystemExit("product not in model (too few interactions)")
    row = m["S"][i_pos[product_id]]
    order = np.argsort(-row.data)[:k]
    return _show(row.indices[order], row.data[order])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int)
    ap.add_argument("--item", type=int)
    ap.add_argument("-k", type=int, default=10)
    a = ap.parse_args()
    print(similar_items(a.item, a.k) if a.item else for_user(a.user, a.k))