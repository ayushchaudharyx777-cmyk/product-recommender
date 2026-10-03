"""Stage 1: item content vectors, retrieval models and candidate generation."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

N_NEIGHBORS = 50
N_KNN, N_SEQ, N_CONTENT, N_POP = 60, 60, 40, 20
FEATURES = ["knn", "knn_norm", "last_sim", "seq", "content", "same_cat", "cat_share", "same_brand",
            "brand_share", "price_log", "price_ratio", "pop_log", "hist_len", "n_sources"]
FEATURE_GROUPS = {
    "collaborative": ["knn", "knn_norm", "last_sim"],
    "sequential": ["seq"],
    "content": ["content", "same_cat", "cat_share", "same_brand", "brand_share", "price_log", "price_ratio"],
    "popularity": ["pop_log"],
}


def content_matrix(meta):
    """L2-normalised sparse item vectors built only from metadata, so they exist for brand-new products.

    Tokens: category id, each level of the category path, brand, price decile."""
    m = meta.reset_index(drop=True)
    rows, toks, vals = [], [], []

    def add(series, prefix, weight):
        s = series.dropna()
        rows.append(s.index.to_numpy()); toks.append(prefix + s.astype(str)); vals.append(np.full(len(s), weight))

    add(m["category_id"], "cid:", 1.5)
    parts = m["category_code"].str.split(".")
    for level in (1, 2, 3):
        add(parts.str[:level].str.join("."), f"c{level}:", 1.0)
    add(m["brand"], "b:", 1.0)
    price = np.log1p(m["price"].fillna(0))
    add(pd.Series(pd.qcut(price, 10, labels=False, duplicates="drop"), index=m.index), "p:", 0.7)
    codes, _ = pd.factorize(pd.concat(toks, ignore_index=True))
    V = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), codes)), shape=(len(m), codes.max() + 1))
    return normalize(V).astype(np.float32)


class ItemFeatures:
    """Metadata of the items in one universe, in item-index order."""

    def __init__(self, meta):
        m = meta.reset_index(drop=True)
        n = len(m)
        cat, cat_names = pd.factorize(m["category_code"])
        self.cat_missing = len(cat_names)
        self.cat = np.where(cat < 0, self.cat_missing, cat)
        self.cat_names = [str(c) for c in cat_names] + ["unknown"]
        brand, brand_names = pd.factorize(m["brand"])
        self.brand_missing = len(brand_names)
        self.brand = np.where(brand < 0, self.brand_missing, brand)
        self.price = m["price"].fillna(0).to_numpy(dtype=np.float64)
        one = np.ones(n, dtype=np.float32)
        self.Ccat = sp.csr_matrix((one, (np.arange(n), self.cat)), shape=(n, self.cat_missing + 1))
        self.Cbrand = sp.csr_matrix((one, (np.arange(n), self.brand)), shape=(n, self.brand_missing + 1))
        self.V = content_matrix(m)


def top_k_sparse(S, k):
    """Keep the k largest values in every row of a sparse matrix."""
    S = S.tocsr()
    rows, cols, vals = [], [], []
    for i in range(S.shape[0]):
        a, b = S.indptr[i], S.indptr[i + 1]
        d, c = S.data[a:b], S.indices[a:b]
        if len(d) > k:
            idx = np.argpartition(d, -k)[-k:]
            d, c = d[idx], c[idx]
        rows.append(np.full(len(d), i)); cols.append(c); vals.append(d)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=S.shape, dtype=np.float32)


class Retriever:
    """Collaborative, sequential and popularity models fitted on one snapshot of user histories.

    `hist` has integer columns u, i plus weight w and first-seen time t."""

    def __init__(self, hist, shape):
        h = hist.sort_values(["u", "t"])
        u, i = h.u.to_numpy(), h.i.to_numpy()
        n_users, n_items = shape
        self.X = sp.csr_matrix((h.w.to_numpy(dtype=np.float32), (u, i)), shape=shape)
        Xn = normalize(self.X, axis=0)
        S = (Xn.T @ Xn).tocsr()
        S = (S - sp.diags(S.diagonal())).tocsr()
        S.eliminate_zeros()
        self.S = top_k_sparse(S, N_NEIGHBORS)                      # item-item cosine neighbours
        a, b = u[:-1] == u[1:], u[:-2] == u[2:]                     # item -> next (1.0), -> one after (0.5)
        T = (sp.coo_matrix((np.ones(a.sum()), (i[:-1][a], i[1:][a])), shape=(n_items, n_items))
             + 0.5 * sp.coo_matrix((np.ones(b.sum()), (i[:-2][b], i[2:][b])), shape=(n_items, n_items)))
        self.T = normalize(T.tocsr().astype(np.float32), norm="l1", axis=1)
        ends = np.flatnonzero(np.r_[u[1:] != u[:-1], True]) if len(u) else np.array([], dtype=int)
        self.last = np.full(n_users, -1, dtype=np.int64)
        self.last[u[ends]] = i[ends]
        self.last2 = np.full(n_users, -1, dtype=np.int64)
        ok = (ends > 0) & (u[np.maximum(ends - 1, 0)] == u[ends])
        self.last2[u[ends[ok]]] = i[ends[ok] - 1]
        self.pop = np.asarray(self.X.sum(0)).ravel()
        self.popn = np.log1p(self.pop) / max(np.log1p(self.pop).max(), 1e-9)
        self.top_pop = np.argsort(-self.pop)[:N_POP]
        self.hist_len = np.diff(self.X.indptr)


def base_scores(ret, feats, Xb, last, last2):
    """Dense score matrices (batch x items) of the three personalised retrieval models."""
    knn = (Xb @ ret.S).toarray()
    seq = ret.T[last].toarray()
    has2 = last2 >= 0
    if has2.any():
        seq[has2] += 0.5 * ret.T[last2[has2]].toarray()
    profile = normalize((Xb > 0).astype(np.float32) @ feats.V)      # mean content vector of the history
    content = np.asarray((feats.V @ profile.T.toarray()).T)
    tie = 1e-4 * ret.popn                                           # popularity only breaks ties
    return {"knn": knn + tie * (knn > 0), "seq": seq + tie * (seq > 0), "content": content + 0.02 * ret.popn}


def candidates(ret, feats, Xb, last, last2):
    """Union of the top items of every retrieval source, with ranking features for each pair."""
    sc = base_scores(ret, feats, Xb, last, last2)
    seen = Xb.toarray() > 0

    def top(s, n):
        s = np.where(seen, 0, s)
        thr = np.partition(s, -n, axis=1)[:, -n][:, None]
        return (s >= thr) & (s > 0)

    m_knn, m_seq, m_con = top(sc["knn"], N_KNN), top(sc["seq"], N_SEQ), top(sc["content"], N_CONTENT)
    mask = m_knn | m_seq | m_con
    mask[:, ret.top_pop] = True
    mask &= ~seen
    r, c = np.nonzero(mask)
    hist_len = np.diff(Xb.indptr)[r]
    Xbin = (Xb > 0).astype(np.float32)
    cat_cnt = (Xbin @ feats.Ccat).toarray()
    brand_cnt = (Xbin @ feats.Cbrand).toarray()
    last_r = last[r]
    last_sim = ret.S[last].toarray()
    knn = sc["knn"][r, c]
    F = np.column_stack([
        knn,
        knn / (sc["knn"].max(1)[r] + 1e-9),
        last_sim[r, c],
        sc["seq"][r, c],
        sc["content"][r, c],
        (feats.cat[c] == feats.cat[last_r]) & (feats.cat[c] != feats.cat_missing),
        cat_cnt[r, feats.cat[c]] / hist_len,
        (feats.brand[c] == feats.brand[last_r]) & (feats.brand[c] != feats.brand_missing),
        brand_cnt[r, feats.brand[c]] / hist_len * (feats.brand[c] != feats.brand_missing),
        np.log1p(feats.price[c]),
        np.log1p(feats.price[c]) - np.log1p(feats.price[last_r]),
        np.log1p(ret.pop[c]),
        hist_len,
        m_knn[r, c].astype(int) + m_seq[r, c] + m_con[r, c],
    ]).astype(np.float32)
    return F, r, c, sc, seen
