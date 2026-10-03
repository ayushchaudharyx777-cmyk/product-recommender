"""Generate a SYNTHETIC event log with the same schema as the Kaggle file, for tests and CI.

Never report numbers from it.  Usage:  python scripts/make_synthetic_data.py data/events.csv
"""
import sys

import numpy as np
import pandas as pd


def make(path, n_users=6000, n_items=2500, n_cats=25, seed=0):
    rng = np.random.default_rng(seed)
    cat = rng.integers(0, n_cats, n_items)
    code = np.array([f"dept{c % 4}.group{c % 9}.type{c}" for c in cat], dtype=object)
    code[rng.random(n_items) < 0.27] = None
    brand = np.array([f"brand{(c * 3 + b) % 40}" for c, b in zip(cat, rng.integers(0, 4, n_items))], dtype=object)
    brand[rng.random(n_items) < 0.24] = None
    price = np.round(np.exp(rng.normal(3 + cat % 5 * 0.5, 0.4)), 2)
    by_cat = [np.flatnonzero(cat == c) for c in range(n_cats)]
    weight = [1 / np.arange(1, len(ix) + 1) ** 0.8 for ix in by_cat]
    weight = [w / w.sum() for w in weight]
    start = pd.Timestamp("2020-09-24")
    rows = []
    for u in range(n_users):
        fav = rng.integers(0, n_cats, 2)
        for v in range(1 + rng.poisson(1.0)):
            t = start + pd.Timedelta(seconds=int(rng.integers(0, 150 * 86400)))
            item = None
            for _ in range(1 + rng.geometric(0.45)):
                c = fav[0] if rng.random() < 0.75 else (fav[1] if rng.random() < 0.6 else rng.integers(0, n_cats))
                ix = by_cat[c]
                if item is not None and cat[item] == c and rng.random() < 0.5:
                    item = ix[(np.searchsorted(ix, item) + 1) % len(ix)]      # "next product" structure
                else:
                    item = rng.choice(ix, p=weight[c])
                events = ["view"]
                if rng.random() < 0.10:
                    events.append("cart")
                    if rng.random() < 0.6:
                        events.append("purchase")
                for e in events:
                    t += pd.Timedelta(seconds=int(rng.integers(5, 120)))
                    rows.append((t, e, 100000 + item, 2144415920000000000 + cat[item], code[item], brand[item],
                                 price[item], 1515915625000000000 + u, f"s{u}_{v}"))
    df = pd.DataFrame(rows, columns=["event_time", "event_type", "product_id", "category_id", "category_code",
                                     "brand", "price", "user_id", "user_session"]).sort_values("event_time")
    df["event_time"] = df["event_time"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/events.csv"
    print(f"wrote {len(make(out)):,} synthetic events to {out}")
