"""Offline evaluation: one protocol runner for every model, plus segment and cold-start analysis."""
import os
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from metrics import beyond_accuracy, hit_matrix, paired_bootstrap, top_k_dense, top_k_pairs, user_metrics
from retrieval import FEATURE_GROUPS, FEATURES, Retriever, candidates, content_matrix

BATCH = 2000
RANKER = "Hybrid ranker (LambdaRank)"
KNN = "Item-item KNN"


def fit_ranker(F, y, groups, cols, fast=False):
    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=100 if fast else 300, learning_rate=0.05,
                           num_leaves=31, min_child_samples=20 if fast else 50, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1,
                           n_jobs=min(8, os.cpu_count() or 1))
    return model.fit(pd.DataFrame(F, columns=FEATURES)[cols], y, group=groups)


def score(model, F, cols):
    return model.predict(pd.DataFrame(F, columns=FEATURES)[cols])


def collect(ret, feats, Y, users, svd=None):
    """Stage 1 for `users`: candidate table (features, labels) and, if svd is given, baseline top-K lists."""
    Fs, ys, rs, cs = [], [], [], []
    tops = {m: [] for m in ["Popularity", "Matrix factorisation (SVD)", KNN, "Sequential transitions",
                            "Content-based"]} if svd else {}
    for s in range(0, len(users), BATCH):
        U = users[s:s + BATCH]
        F, r, c, sc, seen = candidates(ret, feats, ret.X[U], ret.last[U], ret.last2[U])
        Fs.append(F); ys.append((Y[U].toarray() > 0)[r, c]); rs.append(r + s); cs.append(c)
        if svd:
            backfill = 1e-6 * ret.popn                      # items with no signal are ordered by popularity
            tops["Popularity"].append(top_k_dense(np.tile(ret.pop, (len(U), 1)), seen))
            tops["Matrix factorisation (SVD)"].append(top_k_dense(svd[0][U] @ svd[1].T, seen))
            tops[KNN].append(top_k_dense(sc["knn"] + backfill, seen))
            tops["Sequential transitions"].append(top_k_dense(sc["seq"] + backfill, seen))
            tops["Content-based"].append(top_k_dense(sc["content"], seen))
    return SimpleNamespace(F=np.vstack(Fs), y=np.concatenate(ys), r=np.concatenate(rs), c=np.concatenate(cs),
                           tops={m: np.vstack(t) for m, t in tops.items()})


def run_protocol(hist_a, Y_a, hist_b, Y_b, shape, feats, fast=False, ablation=False, log=print):
    """Train the ranker on (history A -> labels A) and evaluate every model on (history B -> labels B).

    Retrieval models are refitted on each history, so nothing from the label period leaks into features."""
    ret_a, ret_b = Retriever(hist_a, shape), Retriever(hist_b, shape)
    train_users = np.flatnonzero((ret_a.hist_len > 0) & (np.diff(Y_a.indptr) > 0))
    test_users = np.flatnonzero((ret_b.hist_len > 0) & (np.diff(Y_b.indptr) > 0))
    log(f"  ranker-train users: {len(train_users):,}   test users: {len(test_users):,}")

    tr = collect(ret_a, feats, Y_a, train_users)
    keep = np.bincount(tr.r, weights=tr.y, minlength=len(train_users))[tr.r] > 0   # label was retrieved
    groups = np.bincount(tr.r[keep])
    groups = groups[groups > 0]
    Xtr, ytr = tr.F[keep], tr.y[keep].astype(int)
    if len(groups) < 10 or len(test_users) < 10:
        raise ValueError("too few users with both a history and later interactions to train or test the ranker")
    log(f"  ranker training set: {len(Xtr):,} candidate rows from {len(groups):,} users")
    ranker = fit_ranker(Xtr, ytr, groups, FEATURES, fast)

    svd = TruncatedSVD(min(64, shape[1] - 1), random_state=42)
    te = collect(ret_b, feats, Y_b, test_users, svd=(svd.fit_transform(ret_b.X), svd.components_.T))
    tops = dict(te.tops)
    tops[RANKER] = top_k_pairs(score(ranker, te.F, FEATURES), te.r, te.c, len(test_users))
    n_rel = np.diff(Y_b[test_users].indptr)
    per_user = {m: user_metrics(hit_matrix(t, Y_b, test_users), n_rel) for m, t in tops.items()}

    out = SimpleNamespace(ranker=ranker, ret_b=ret_b, test_users=test_users, per_user=per_user, tops=tops)
    out.summary = {
        "n_train_users": int(len(groups)), "n_test_users": int(len(test_users)),
        "relevant_items_per_test_user": round(float(n_rel.mean()), 2),
        "candidates_per_user": round(len(te.y) / len(test_users), 1),
        "candidate_recall": round(float(te.y.sum() / n_rel.sum()), 4),
        "models": {m: d.mean().round(4).to_dict() for m, d in per_user.items()},
        "beyond_accuracy": {m: beyond_accuracy(t, feats.cat, ret_b.pop) for m, t in tops.items()},
        "ranker_vs_knn_ndcg@10": paired_bootstrap(per_user[RANKER]["ndcg@10"], per_user[KNN]["ndcg@10"]),
    }
    imp = pd.Series(ranker.booster_.feature_importance("gain"), index=FEATURES)
    out.summary["feature_importance_pct"] = (100 * imp / imp.sum()).round(1).sort_values(ascending=False).to_dict()
    if ablation:
        full = out.summary["models"][RANKER]
        out.summary["ablation"] = {"all features": {"ndcg@10": full["ndcg@10"], "recall@10": full["recall@10"]}}
        for name, drop in FEATURE_GROUPS.items():
            cols = [f for f in FEATURES if f not in drop]
            model = fit_ranker(Xtr, ytr, groups, cols, fast)
            top = top_k_pairs(score(model, te.F, cols), te.r, te.c, len(test_users))
            m = user_metrics(hit_matrix(top, Y_b, test_users), n_rel).mean()
            out.summary["ablation"][f"without {name}"] = {"ndcg@10": round(float(m["ndcg@10"]), 4),
                                                           "recall@10": round(float(m["recall@10"]), 4)}
    return out


def segment_table(hist_b, res, feats):
    """Ranker vs KNN quality by user segment: history length, shopper type, price tier, favourite category."""
    h = hist_b[hist_b.u.isin(res.test_users)].assign(cat=lambda d: feats.cat[d.i.to_numpy()],
                                                      price=lambda d: feats.price[d.i.to_numpy()])
    g = h.groupby("u")
    fav = (h.groupby(["u", "cat"]).size().reset_index(name="n").sort_values("n")
            .drop_duplicates("u", keep="last").set_index("u")["cat"])
    prof = pd.DataFrame({"n": g.size(), "buyer": g["bought"].max(), "price": g["price"].median(), "cat": fav})
    prof = prof.loc[res.test_users]
    top_cats = prof["cat"].value_counts().index[:5]
    seg = {
        "History length": pd.cut(prof["n"], [0, 2, 4, 9, np.inf], labels=["1-2 products", "3-4 products",
                                                                           "5-9 products", "10+ products"]),
        "Shopper type": np.where(prof["buyer"], "has purchased", "browser only"),
        "Price tier": pd.cut(prof["price"].rank(pct=True), [0, 1 / 3, 2 / 3, 1],
                             labels=["budget", "mid-range", "premium"]),
        "Favourite category": prof["cat"].map(lambda c: feats.cat_names[c].split(".")[-1]
                                              if c in top_cats else "other categories"),
    }
    rk, kn = res.per_user[RANKER], res.per_user[KNN]
    rows = []
    for kind, labels in seg.items():
        labels = np.asarray(labels, dtype=object)
        for name in pd.unique(labels):
            if pd.isna(name):
                continue
            m = labels == name
            rows.append({"segment_type": kind, "segment": name, "users": int(m.sum()),
                         "precision@10": rk["precision@10"][m].mean(), "recall@10": rk["recall@10"][m].mean(),
                         "ndcg@10": rk["ndcg@10"][m].mean(), "knn_ndcg@10": kn["ndcg@10"][m].mean()})
    out = pd.DataFrame(rows)
    out["lift_vs_knn_pct"] = 100 * (out["ndcg@10"] / out["knn_ndcg@10"].replace(0, np.nan) - 1)
    return out.sort_values(["segment_type", "users"], ascending=[True, False]).round(4).reset_index(drop=True)


def cold_start_eval(ui, model_users, model_items, meta_all, n_sample=5000, seed=42):
    """Content-based fallback for users outside the collaborative model, scored over the FULL catalogue.

    Each sampled cold user's last product is hidden and predicted from the earlier ones using metadata only."""
    cold = ui[~ui.user_id.isin(model_users)]
    n_items = cold.groupby("user_id")["product_id"].transform("size")
    eligible = cold.loc[n_items >= 2, "user_id"].unique()
    rng = np.random.default_rng(seed)
    sample = rng.choice(eligible, size=min(n_sample, len(eligible)), replace=False)
    c = cold[cold.user_id.isin(sample)].sort_values(["user_id", "t"])
    all_items = meta_all.index.to_numpy()
    u = pd.factorize(c.user_id)[0]
    i = pd.Index(all_items).get_indexer(c.product_id)
    last = (c.groupby("user_id").cumcount(ascending=False) == 0).to_numpy()
    n = u.max() + 1
    target = np.zeros(n, dtype=np.int64)
    target[u[last]] = i[last]
    H = sp.csr_matrix((np.ones((~last).sum(), dtype=np.float32), (u[~last], i[~last])), shape=(n, len(all_items)))

    V = content_matrix(meta_all)
    pop = np.bincount(pd.Index(all_items).get_indexer(ui.product_id), minlength=len(all_items)).astype(float)
    pop -= np.bincount(target, minlength=len(all_items))            # the hidden interactions are not counted
    popn = np.log1p(pop) / np.log1p(pop).max()
    profile = normalize(H @ V)
    res = {"Popularity": np.zeros(2), "Content-based fallback": np.zeros(2)}
    for s in range(0, n, 500):
        seen = H[s:s + 500].toarray() > 0
        t = target[s:s + 500, None]
        content = np.asarray((V @ profile[s:s + 500].T.toarray()).T) + 0.02 * popn
        for name, sc in (("Popularity", np.tile(popn, (len(t), 1))), ("Content-based fallback", content)):
            top = top_k_dense(sc, seen, 10)
            pos = np.where(top == t)[1]
            res[name] += [len(pos), (1 / np.log2(pos + 2)).sum()]
    return {"n_users": int(n), "catalogue_size": int(len(all_items)),
            "eligible_cold_users": int(len(eligible)),
            "targets_outside_cf_model_pct": round(100 * float((~np.isin(all_items[target], model_items)).mean()), 1),
            "models": {m: {"hit@10": round(v[0] / n, 4), "ndcg@10": round(v[1] / n, 4)} for m, v in res.items()}}
