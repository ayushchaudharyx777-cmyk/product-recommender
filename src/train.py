"""Train + evaluate recommenders on REES46 electronics events.
Run from project root:  python src/train.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

DATA = Path("data/events.csv")
OUT = Path("models")
WEIGHTS = {"view": 1.0, "cart": 3.0, "purchase": 5.0}
MIN_USER_ITEMS, MIN_ITEM_USERS = 3, 5
K, TOPK_NEIGHBORS, SVD_DIM, SEED = 10, 50, 64, 42


def load():
    df = pd.read_csv(DATA).drop_duplicates()
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    df["w"] = df["event_type"].map(WEIGHTS)
    return df.dropna(subset=["w"])


def build_interactions(df):
    """One row per (user, item): summed weight + first time seen."""
    ui = (df.groupby(["user_id", "product_id"])
            .agg(w=("w", "sum"), t=("event_time", "min")).reset_index())
    ui["w"] = np.log1p(ui["w"].clip(upper=20))
    while True:  # iterative k-core filter
        n = len(ui)
        ui = ui[ui.groupby("product_id")["user_id"].transform("size") >= MIN_ITEM_USERS]
        ui = ui[ui.groupby("user_id")["product_id"].transform("size") >= MIN_USER_ITEMS]
        if len(ui) == n:
            return ui.reset_index(drop=True)


def split(ui):
    """Leave-last-out: each user's most recently discovered item is the test item."""
    ui = ui.sort_values(["user_id", "t"])
    is_last = ui.groupby("user_id").cumcount(ascending=False) == 0
    return ui[~is_last], ui[is_last]


def top_k_sparse(S, k):
    S = S.tocsr()
    rows, cols, vals = [], [], []
    for i in range(S.shape[0]):
        s, e = S.indptr[i], S.indptr[i + 1]
        d, c = S.data[s:e], S.indices[s:e]
        if len(d) > k:
            idx = np.argpartition(d, -k)[-k:]
            d, c = d[idx], c[idx]
        rows.append(np.full(len(d), i)); cols.append(c); vals.append(d)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=S.shape, dtype=np.float32)


def evaluate(score_fn, X, test_u, test_i, n_items, batch=2000):
    hits = ndcg = mrr = 0.0
    seen_items = set()
    for s in range(0, len(test_u), batch):
        u, y = test_u[s:s + batch], test_i[s:s + batch]
        sc = np.asarray(score_fn(u), dtype=np.float32)
        sc[X[u].nonzero()] = -np.inf  # never recommend already-seen items
        top = np.argpartition(-sc, K, axis=1)[:, :K]
        order = np.argsort(-np.take_along_axis(sc, top, 1), axis=1)
        top = np.take_along_axis(top, order, 1)
        seen_items.update(top.ravel().tolist())
        r, pos = np.where(top == y[:, None])
        hits += len(r); ndcg += (1 / np.log2(pos + 2)).sum(); mrr += (1 / (pos + 1)).sum()
    n = len(test_u)
    return {f"hit@{K}": round(hits / n, 4), f"ndcg@{K}": round(ndcg / n, 4),
            f"mrr@{K}": round(mrr / n, 4), "coverage": round(len(seen_items) / n_items, 4)}


def main():
    OUT.mkdir(exist_ok=True)
    df = load()
    ui = build_interactions(df)
    users, items = np.sort(ui.user_id.unique()), np.sort(ui.product_id.unique())
    u_idx = pd.Series(np.arange(len(users)), index=users)
    i_idx = pd.Series(np.arange(len(items)), index=items)
    train, test = split(ui)
    shape = (len(users), len(items))
    X = sp.csr_matrix((train.w.values.astype(np.float32),
                       (u_idx[train.user_id].values, i_idx[train.product_id].values)), shape=shape)
    test_u, test_i = u_idx[test.user_id].values, i_idx[test.product_id].values
    print(f"users={shape[0]:,} items={shape[1]:,} train={len(train):,} test={len(test):,} "
          f"density={X.nnz / (shape[0] * shape[1]):.5f}")

    # 1) popularity baseline
    pop = np.asarray(X.sum(0)).ravel()
    # 2) item-item cosine KNN
    Xn = normalize(X, axis=0)
    S = (Xn.T @ Xn).tolil(); S.setdiag(0)
    S = top_k_sparse(S.tocsr(), TOPK_NEIGHBORS)
    # 3) SVD matrix factorisation
    svd = TruncatedSVD(SVD_DIM, random_state=SEED)
    U = svd.fit_transform(X).astype(np.float32)
    V = svd.components_.T.astype(np.float32)

    models = {
        "popularity": lambda u: np.tile(pop, (len(u), 1)),
        "item_knn": lambda u: (X[u] @ S).toarray(),
        "svd": lambda u: U[u] @ V.T,
    }
    metrics = {name: evaluate(fn, X, test_u, test_i, shape[1]) for name, fn in models.items()}
    print(pd.DataFrame(metrics).T.to_string())
    best = max(metrics, key=lambda m: metrics[m][f"ndcg@{K}"])
    print("best:", best)

    # refit on ALL interactions for serving
    Xf = sp.csr_matrix((ui.w.values.astype(np.float32),
                        (u_idx[ui.user_id].values, i_idx[ui.product_id].values)), shape=shape)
    Xfn = normalize(Xf, axis=0)
    Sf = (Xfn.T @ Xfn).tolil(); Sf.setdiag(0)
    Sf = top_k_sparse(Sf.tocsr(), TOPK_NEIGHBORS)
    meta = (df.sort_values("event_time").drop_duplicates("product_id", keep="last")
              .set_index("product_id")[["category_code", "brand", "price"]].reindex(items))
    joblib.dump({"X": Xf, "S": Sf, "pop": np.asarray(Xf.sum(0)).ravel(),
                 "users": users, "items": items, "meta": meta}, OUT / "recommender.joblib")
    json.dump({"best": best, "metrics": metrics, "n_users": shape[0], "n_items": shape[1]},
              open(OUT / "metrics.json", "w"), indent=2)
    print("saved -> models/recommender.joblib, models/metrics.json")


if __name__ == "__main__":
    main()