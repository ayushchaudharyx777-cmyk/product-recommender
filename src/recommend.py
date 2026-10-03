"""Serve recommendations from the hybrid retrieve-and-rank model.
Usage:
  python src/recommend.py --user 1515915625353230683
  python src/recommend.py --item 1821813
"""
import argparse
import types

import joblib
import numpy as np
import pandas as pd

from hybrid_ranker import FEATURES, candidates

a = joblib.load("models/ranker.joblib")
p, m, ranker, info = types.SimpleNamespace(**a["phase"]), a["meta"], a["ranker"], a["info"]
u_pos = {u: i for i, u in enumerate(a["users"])}
i_pos = {x: i for i, x in enumerate(a["items"])}


def _show(idx, scores):
    out = info.iloc[idx].copy()
    out["score"] = np.round(scores, 4)
    return out


def for_user(user_id, k=10):
    """Retrieve candidates, re-rank with LightGBM. Unknown users get popular items."""
    if user_id not in u_pos:
        idx = np.argsort(-p.pop)[:k]
        return _show(idx, p.pop[idx])
    f, _, _, c = candidates(p, np.array([u_pos[user_id]]), m)
    score = ranker.predict(pd.DataFrame(f, columns=FEATURES))
    order = np.argsort(-score)[:k]
    return _show(c[order], score[order])


def similar_items(product_id, k=10):
    """Item-item cosine neighbours."""
    if product_id not in i_pos:
        raise SystemExit("product not in model (too few interactions)")
    row = p.S[i_pos[product_id]]
    order = np.argsort(-row.data)[:k]
    return _show(row.indices[order], row.data[order])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int)
    ap.add_argument("--item", type=int)
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args()
    print(similar_items(args.item, args.k) if args.item else for_user(args.user, args.k))