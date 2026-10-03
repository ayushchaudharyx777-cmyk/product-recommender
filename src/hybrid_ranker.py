"""Hybrid retrieve-and-rank recommender: candidate retrieval -> LightGBM LambdaRank re-ranker.

Stage 1 pulls ~100 candidates per user from three sources (item-KNN, sequential
"what do people open next" transitions, popularity). Stage 2 re-ranks them with a
learning-to-rank model using collaborative, sequential and content features.

Run from project root (after train.py):  python src/hybrid_ranker.py
"""
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from train import OUT, TOPK_NEIGHBORS, build_interactions, load, top_k_sparse

K, BATCH = 10, 2000
N_KNN, N_SEQ, N_POP = 60, 60, 20
FEATURES = ["knn", "knn_norm", "seq", "last_sim", "pop_log", "hist_len", "same_cat",
            "same_brand", "price_log", "price_ratio", "cat_share"]


class Phase:
    """Everything stage 1 needs, built from one snapshot of user histories."""

    def __init__(self, hist, target, shape):
        h = hist.sort_values(["u", "t"])
        u, i = h.u.values, h.i.values
        self.X = sp.csr_matrix((h.w.values.astype(np.float32), (u, i)), shape=shape)
        Xn = normalize(self.X, axis=0)
        S = (Xn.T @ Xn).tolil(); S.setdiag(0)
        self.S = top_k_sparse(S.tocsr(), TOPK_NEIGHBORS)
        # sequential transitions: item -> next item (1.0), item -> item after next (0.5)
        a, b = u[:-1] == u[1:], u[:-2] == u[2:]
        n = shape[1]
        T = (sp.coo_matrix((np.ones(a.sum()), (i[:-1][a], i[1:][a])), shape=(n, n))
             + 0.5 * sp.coo_matrix((np.ones(b.sum()), (i[:-2][b], i[2:][b])), shape=(n, n)))
        self.T = normalize(T.tocsr().astype(np.float32), norm="l1", axis=1)
        ends = np.flatnonzero(np.r_[u[1:] != u[:-1], True])
        self.last = np.zeros(shape[0], dtype=np.int64); self.last[u[ends]] = i[ends]
        self.last2 = np.full(shape[0], -1, dtype=np.int64)
        ok = (ends > 0) & (u[np.maximum(ends - 1, 0)] == u[ends])
        self.last2[u[ends[ok]]] = i[ends[ok] - 1]
        self.pop = np.asarray(self.X.sum(0)).ravel()
        self.top_pop = np.argsort(-self.pop)[:N_POP]
        self.hist_len = np.diff(self.X.indptr)
        self.target = target


def candidates(p, U, m):
    """Candidate (user, item) pairs for a batch of users, with features and labels."""
    Xb = p.X[U]
    seen = Xb.toarray() > 0
    knn = (Xb @ p.S).toarray()
    seq = p.T[p.last[U]].toarray()
    has2 = p.last2[U] >= 0
    seq[has2] += 0.5 * p.T[p.last2[U][has2]].toarray()

    def top(sc, n):
        sc = np.where(seen, 0, sc)
        thr = np.partition(sc, -n, axis=1)[:, -n][:, None]
        return (sc >= thr) & (sc > 0)

    mask = top(knn, N_KNN) | top(seq, N_SEQ)
    mask[:, p.top_pop] = True
    mask &= ~seen
    r, c = np.nonzero(mask)
    last = p.last[U][r]
    last_sim = p.S[p.last[U]].toarray()
    cat_cnt = ((Xb > 0).astype(np.float32) @ m["C"]).toarray()
    hist_len = p.hist_len[U][r]
    f = np.column_stack([
        knn[r, c],
        knn[r, c] / (knn.max(1)[r] + 1e-9),
        seq[r, c],
        last_sim[r, c],
        np.log1p(p.pop[c]),
        hist_len,
        (m["cat"][c] == m["cat"][last]) & (m["cat"][c] != m["cat_missing"]),
        (m["brand"][c] == m["brand"][last]) & (m["brand"][c] >= 0),
        np.log1p(m["price"][c]),
        np.log1p(m["price"][c]) - np.log1p(m["price"][last]),
        cat_cnt[r, m["cat"][c]] / hist_len,
    ]).astype(np.float32)
    return f, c == p.target[U][r], r, c


def topk_stats(score, y, r):
    """hits, ndcg sum, mrr sum for top-K ranking of each user's candidates."""
    order = np.lexsort((-score, r))
    r_s, y_s = r[order], y[order]
    start = np.r_[0, np.flatnonzero(r_s[1:] != r_s[:-1]) + 1]
    pos = np.arange(len(r_s)) - np.repeat(start, np.diff(np.r_[start, len(r_s)]))
    hit = y_s & (pos < K)
    return np.array([hit.sum(), (1 / np.log2(pos[hit] + 2)).sum(), (1 / (pos[hit] + 1)).sum()])


def main():
    df = load()
    ui = build_interactions(df)
    users, items = np.sort(ui.user_id.unique()), np.sort(ui.product_id.unique())
    shape = (len(users), len(items))
    ui["u"] = pd.Series(np.arange(shape[0]), index=users)[ui.user_id].values
    ui["i"] = pd.Series(np.arange(shape[1]), index=items)[ui.product_id].values
    ui = ui.sort_values(["u", "t"])
    back = ui.groupby("u").cumcount(ascending=False).values  # 0 = last item, 1 = second last

    def target_of(k):
        t = np.zeros(shape[0], dtype=np.int64)
        t[ui.u.values[back == k]] = ui.i.values[back == k]
        return t

    # item metadata for content features
    meta = (df.sort_values("event_time").drop_duplicates("product_id", keep="last")
              .set_index("product_id")[["category_code", "brand", "price"]].reindex(items))
    cat = pd.factorize(meta.category_code)[0]
    cat_missing = cat.max() + 1
    cat = np.where(cat < 0, cat_missing, cat)
    m = {"cat": cat, "cat_missing": cat_missing, "brand": pd.factorize(meta.brand)[0],
         "price": meta.price.fillna(0).values,
         "C": sp.csr_matrix((np.ones(shape[1], dtype=np.float32), (np.arange(shape[1]), cat)),
                            shape=(shape[1], cat_missing + 1))}

    # Phase A: history without the last two items, label = second-last item (trains the ranker)
    # Phase B: history without the last item, label = last item (same test split as train.py)
    pa = Phase(ui[back >= 2], target_of(1), shape)
    pb = Phase(ui[back >= 1], target_of(0), shape)

    Xs, ys, groups = [], [], []
    for s in range(0, shape[0], BATCH):
        U = np.arange(s, min(s + BATCH, shape[0]))
        f, y, r, _ = candidates(pa, U, m)
        keep = (np.bincount(r, weights=y, minlength=len(U)) > 0)[r]  # users whose label was retrieved
        Xs.append(f[keep]); ys.append(y[keep]); groups.append(np.bincount(r[keep], minlength=len(U)))
    Xtr, ytr = np.vstack(Xs), np.concatenate(ys).astype(int)
    g = np.concatenate(groups); g = g[g > 0]
    print(f"ranker training set: {len(Xtr):,} candidate rows, {len(g):,} users")
    ranker = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05,
                            num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.8, random_state=42, verbose=-1)
    ranker.fit(pd.DataFrame(Xtr, columns=FEATURES), ytr, group=g)

    names = {"item_knn only": 0, "sequence only": 2}
    stats = {k: np.zeros(3) for k in [*names, "hybrid ranker"]}
    found = n_cand = 0
    for s in range(0, shape[0], BATCH):
        U = np.arange(s, min(s + BATCH, shape[0]))
        f, y, r, _ = candidates(pb, U, m)
        found += y.sum(); n_cand += len(y)
        for k, col in names.items():
            stats[k] += topk_stats(f[:, col], y, r)
        stats["hybrid ranker"] += topk_stats(ranker.predict(pd.DataFrame(f, columns=FEATURES)), y, r)

    res = pd.DataFrame(stats, index=[f"hit@{K}", f"ndcg@{K}", f"mrr@{K}"]).T / shape[0]
    print(f"\ncandidates per user: {n_cand / shape[0]:.0f}   candidate recall: {found / shape[0]:.4f}")
    print(res.round(4).to_string())
    imp = pd.Series(ranker.booster_.feature_importance("gain"), index=FEATURES)
    print("\nfeature importance (gain %):")
    print((100 * imp / imp.sum()).sort_values(ascending=False).round(1).to_string())

        # refit retrieval on ALL interactions for serving (plain dicts so recommend.py can load them)
    full = Phase(ui, np.full(shape[0], -1), shape)
    joblib.dump({"ranker": ranker, "phase": vars(full), "meta": m, "users": users, "items": items,
                 "info": meta[["category_code", "brand", "price"]]}, OUT / "ranker.joblib")
    json.dump({"candidate_recall": round(float(found / shape[0]), 4),
               "candidates_per_user": round(n_cand / shape[0], 1),
               "metrics": res.round(4).to_dict("index"),
               "feature_importance_pct": (100 * imp / imp.sum()).round(1).to_dict()},
              open(OUT / "metrics_ranker.json", "w"), indent=2)
    print("\nsaved -> models/ranker.joblib, models/metrics_ranker.json")


if __name__ == "__main__":
    main()
