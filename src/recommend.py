"""Serving: hybrid ranker for known users and sessions, content-based fallback for everything else."""
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from retrieval import FEATURES, candidates

REASON_OF = {"knn": "cf", "knn_norm": "cf", "last_sim": "cf", "seq": "seq", "content": "content",
             "same_cat": "content", "cat_share": "content", "same_brand": "brand", "brand_share": "brand",
             "price_log": "price", "price_ratio": "price", "pop_log": "pop"}


class Recommender:
    def __init__(self, path="models/recommender.joblib"):
        a = joblib.load(path)
        self.ranker, self.ret, self.feats = a["ranker"], a["ret"], a["feats"]
        self.users, self.items, self.all_items = a["users"], a["items"], a["all_items"]
        self.V_all, self.info_all, self.popn_all = a["V_all"], a["info_all"], a["popn_all"]
        self.u_pos = {int(u): k for k, u in enumerate(self.users)}
        self.i_pos = {int(p): k for k, p in enumerate(self.items)}
        self.a_pos = {int(p): k for k, p in enumerate(self.all_items)}

    # ---------- helpers
    def name(self, pid):
        row = self.info_all.loc[pid]
        brand = row["brand"] if pd.notna(row["brand"]) else ""
        cat = row["category_code"].split(".")[-1] if pd.notna(row["category_code"]) else "product"
        return f"{brand} {cat}".strip() + f" #{pid}"

    def _frame(self, pids, scores, source, why):
        out = self.info_all.loc[pids].reset_index()
        out.columns = ["product_id", "category_code", "brand", "price"]
        out["score"] = np.round(np.asarray(scores, dtype=float), 4)
        out["source"], out["why"] = source, why
        return out

    def popular(self, k=10, source="popularity"):
        idx = np.argsort(-self.ret.pop)[:k]
        return self._frame(self.items[idx], self.ret.popn[idx], source, "popular product")

    def history(self, user_id):
        if int(user_id) not in self.u_pos:
            return self._frame([], [], "history", "")
        row = self.ret.X[self.u_pos[int(user_id)]]
        return self._frame(self.items[row.indices], row.data, "history", "")

    # ---------- stage 1 + stage 2 with explanations
    def _rank(self, Xb, last, last2, k):
        F, _, c, _, _ = candidates(self.ret, self.feats, Xb, np.array([last]), np.array([last2]))
        X = pd.DataFrame(F, columns=FEATURES)
        score = self.ranker.predict(X)
        order = np.argsort(-score)[:k]
        contrib = self.ranker.predict(X.iloc[order], pred_contrib=True)[:, :-1]
        hist = Xb.indices
        sim = self.ret.S[hist][:, c[order]].toarray()
        why = []
        for j, item in enumerate(c[order]):
            groups = {}
            for f, v in zip(FEATURES, contrib[j]):
                if f in REASON_OF:
                    groups[REASON_OF[f]] = groups.get(REASON_OF[f], 0.0) + v
            reasons = []
            for g, v in sorted(groups.items(), key=lambda kv: -kv[1])[:2]:
                if v <= 0:
                    continue
                pid = self.items[item]
                row = self.info_all.loc[pid]
                if g == "cf":
                    anchor = hist[sim[:, j].argmax()] if sim[:, j].max() > 0 else None
                    reasons.append(f"viewed together with {self.name(self.items[anchor])}" if anchor is not None
                                   else "liked by users with a similar history")
                elif g == "seq":
                    reasons.append(f"often opened right after {self.name(self.items[last])}")
                elif g == "content":
                    reasons.append(f"matches your interest in {row['category_code'].split('.')[-1]}"
                                   if pd.notna(row["category_code"]) else "similar to products you viewed")
                elif g == "brand":
                    reasons.append(f"a brand you looked at ({row['brand']})" if pd.notna(row["brand"])
                                   else "similar to products you viewed")
                elif g == "price":
                    reasons.append("in your price range")
                else:
                    reasons.append("popular product")
            why.append("; ".join(dict.fromkeys(reasons)) or "best overall match")
        return self._frame(self.items[c[order]], score[order], "hybrid ranker", why)

    def _content(self, rows, k, source):
        profile = normalize(sp.csr_matrix(self.V_all[rows].sum(0)))
        sc = np.asarray((self.V_all @ profile.T).toarray()).ravel() + 0.02 * self.popn_all
        sc[rows] = -np.inf
        idx = np.argsort(-sc)[:k]
        return self._frame(self.all_items[idx], sc[idx], source, "same category, brand or price band (metadata only)")

    # ---------- public API
    def for_user(self, user_id, k=10):
        u = self.u_pos.get(int(user_id))
        if u is None:
            return self.popular(k, "popularity (unknown user)")
        return self._rank(self.ret.X[u], self.ret.last[u], self.ret.last2[u], k)

    def for_session(self, product_ids, k=10):
        """Recommend from an ordered list of viewed products; works for users the model has never seen."""
        ids = list(dict.fromkeys(int(p) for p in product_ids))
        known = [self.i_pos[p] for p in ids if p in self.i_pos]
        if known:
            Xb = sp.csr_matrix((np.full(len(known), np.log1p(1.0), dtype=np.float32),
                                (np.zeros(len(known), dtype=int), known)), shape=(1, len(self.items)))
            return self._rank(Xb, known[-1], known[-2] if len(known) > 1 else -1, k)
        rows = [self.a_pos[p] for p in ids if p in self.a_pos]
        if rows:
            return self._content(rows, k, "content fallback (new products)")
        return self.popular(k, "popularity (unknown products)")

    def similar(self, product_id, k=10):
        """Collaborative neighbours if the product is in the model, otherwise content-based neighbours."""
        pid = int(product_id)
        if pid in self.i_pos and self.ret.S[self.i_pos[pid]].nnz:
            row = self.ret.S[self.i_pos[pid]]
            order = np.argsort(-row.data)[:k]
            return self._frame(self.items[row.indices[order]], row.data[order], "collaborative",
                               "viewed by the same users")
        if pid in self.a_pos:
            return self._content([self.a_pos[pid]], k, "content fallback (new product)")
        raise KeyError(f"product {pid} is not in the catalogue")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int)
    ap.add_argument("--item", type=int)
    ap.add_argument("--session", type=int, nargs="+")
    ap.add_argument("-k", type=int, default=10)
    a = ap.parse_args()
    rec = Recommender()
    pd.set_option("display.width", 220, "display.max_colwidth", 70)
    print(rec.similar(a.item, a.k) if a.item else rec.for_session(a.session, a.k) if a.session
          else rec.for_user(a.user, a.k))
