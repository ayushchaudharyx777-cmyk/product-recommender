"""Loading, implicit-feedback weighting, k-core filtering and index helpers."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

WEIGHTS = {"view": 1.0, "cart": 3.0, "purchase": 5.0}
MIN_USER_ITEMS, MIN_ITEM_USERS = 3, 5


def load(path):
    """Read the raw event log and attach an implicit-feedback weight to every event."""
    df = pd.read_csv(path).drop_duplicates()
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    df["w"] = df["event_type"].map(WEIGHTS)
    return df.dropna(subset=["w"]).reset_index(drop=True)


def interactions(df):
    """One row per (user, product): log-scaled summed weight, first time seen, bought flag."""
    df = df.assign(bought=df["event_type"].eq("purchase"))
    ui = (df.groupby(["user_id", "product_id"])
            .agg(w=("w", "sum"), t=("event_time", "min"), bought=("bought", "max")).reset_index())
    ui["w"] = np.log1p(ui["w"].clip(upper=20)).astype(np.float32)
    return ui


def kcore(ui, min_user=MIN_USER_ITEMS, min_item=MIN_ITEM_USERS):
    """Iteratively keep users with >= min_user products and products with >= min_item users."""
    while True:
        n = len(ui)
        ui = ui[ui.groupby("product_id")["user_id"].transform("size") >= min_item]
        ui = ui[ui.groupby("user_id")["product_id"].transform("size") >= min_user]
        if len(ui) == n:
            return np.sort(ui.user_id.unique()), np.sort(ui.product_id.unique())


def item_meta(df):
    """Latest known category, brand and price for every product in the catalogue."""
    return (df.sort_values("event_time").drop_duplicates("product_id", keep="last")
              .set_index("product_id")[["category_id", "category_code", "brand", "price"]].sort_index())


def to_index(ui, users, items):
    """Keep interactions inside the (users, items) universe and add integer positions u, i."""
    u = pd.Index(users).get_indexer(ui.user_id)
    i = pd.Index(items).get_indexer(ui.product_id)
    keep = (u >= 0) & (i >= 0)
    out = ui.loc[keep, ["w", "t", "bought"]].copy()
    out["u"], out["i"] = u[keep], i[keep]
    return out.sort_values(["u", "t"]).reset_index(drop=True)


def new_pairs(later, history, n_items):
    """Rows of `later` whose (user, item) pair does not already appear in `history`."""
    seen = history.u.to_numpy() * n_items + history.i.to_numpy()
    key = later.u.to_numpy() * n_items + later.i.to_numpy()
    return later[~np.isin(key, seen)]


def label_matrix(pairs, shape):
    return sp.csr_matrix((np.ones(len(pairs), dtype=np.float32), (pairs.u.to_numpy(), pairs.i.to_numpy())),
                         shape=shape)
