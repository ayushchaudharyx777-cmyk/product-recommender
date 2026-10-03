"""Ranking metrics (Precision@K, Recall@K, NDCG@K, Hit@K) and beyond-accuracy measures."""
import numpy as np
import pandas as pd

KS = (5, 10, 20)
MAX_K = max(KS)


def top_k_dense(scores, seen, k=MAX_K):
    """Top-k item indices per row of a dense score matrix, best first; -1 where nothing is left."""
    sc = np.where(seen, -np.inf, scores)
    idx = np.argpartition(-sc, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(sc, idx, 1), axis=1, kind="stable")
    idx = np.take_along_axis(idx, order, 1)
    idx[np.take_along_axis(sc, idx, 1) == -np.inf] = -1
    return idx


def top_k_pairs(score, r, c, n_rows, k=MAX_K):
    """Top-k items per row from (row, item, score) triples."""
    out = np.full((n_rows, k), -1, dtype=np.int64)
    order = np.lexsort((-score, r))
    r_s, c_s = r[order], c[order]
    start = np.r_[0, np.flatnonzero(r_s[1:] != r_s[:-1]) + 1]
    pos = np.arange(len(r_s)) - np.repeat(start, np.diff(np.r_[start, len(r_s)]))
    keep = pos < k
    out[r_s[keep], pos[keep]] = c_s[keep]
    return out


def hit_matrix(topk, Y, users, batch=4000):
    """Boolean matrix: is the item at each rank one of the user's relevant items?"""
    out = np.zeros(topk.shape, dtype=bool)
    for s in range(0, len(users), batch):
        Yd = Y[users[s:s + batch]].toarray() > 0
        tk = topk[s:s + batch]
        out[s:s + batch] = np.take_along_axis(Yd, np.maximum(tk, 0), 1) & (tk >= 0)
    return out


def user_metrics(hits, n_rel, ks=KS):
    """Per-user precision, recall, NDCG and hit rate at every K."""
    disc = 1.0 / np.log2(np.arange(2, hits.shape[1] + 2))
    ideal = np.cumsum(disc)
    out = {}
    for k in ks:
        h = hits[:, :k]
        n_hit = h.sum(1)
        out[f"precision@{k}"] = n_hit / k
        out[f"recall@{k}"] = n_hit / n_rel
        out[f"ndcg@{k}"] = (h * disc[:k]).sum(1) / ideal[np.minimum(n_rel, k).astype(int) - 1]
        out[f"hit@{k}"] = (n_hit > 0).astype(float)
    return pd.DataFrame(out)


def beyond_accuracy(topk, cat, pop, k=10):
    """Catalogue coverage, intra-list category diversity and novelty of the top-k lists."""
    tk = topk[:, :k]
    valid = tk >= 0
    cats = np.where(valid, cat[np.maximum(tk, 0)], -1 - np.arange(k))        # padding never matches
    same = (cats[:, :, None] == cats[:, None, :]).sum((1, 2)) - k
    share = pop / pop.sum()
    info = -np.log2(np.maximum(share[np.maximum(tk, 0)], 1e-12))
    return {"coverage": round(float(len(np.unique(tk[valid])) / len(pop)), 4),
            "diversity": round(float(1 - (same / (k * (k - 1))).mean()), 4),
            "novelty": round(float(info[valid].mean()), 2)}


def paired_bootstrap(a, b, n_boot=2000, seed=42):
    """Mean difference a - b over users with a 95% bootstrap interval."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"diff": round(float(d.mean()), 4), "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
            "significant": bool(lo > 0 or hi < 0)}
