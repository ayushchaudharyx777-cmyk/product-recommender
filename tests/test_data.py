import numpy as np
import pandas as pd

from data import interactions, kcore, load, new_pairs, to_index


def events(rows):
    df = pd.DataFrame(rows, columns=["event_time", "event_type", "product_id", "user_id"])
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    df["w"] = df["event_type"].map({"view": 1.0, "cart": 3.0, "purchase": 5.0})
    return df


def test_weights_are_summed_capped_and_logged():
    df = events([("2021-01-01", "view", 1, 7), ("2021-01-02", "purchase", 1, 7)] +
                [("2021-01-03", "purchase", 2, 7)] * 10)
    ui = interactions(df).set_index("product_id")
    assert np.isclose(ui.loc[1, "w"], np.log1p(6))
    assert np.isclose(ui.loc[2, "w"], np.log1p(20))          # capped at 20
    assert ui.loc[1, "bought"] and str(ui.loc[1, "t"].date()) == "2021-01-01"


def test_kcore_keeps_only_dense_users_and_items():
    rows = [("2021-01-01", "view", p, u) for u in range(6) for p in range(4)] + [("2021-01-01", "view", 99, 0)]
    users, items = kcore(interactions(events(rows)), min_user=3, min_item=5)
    assert 99 not in items and len(users) == 6 and len(items) == 4


def test_new_pairs_removes_products_already_in_history():
    hist = pd.DataFrame({"u": [0, 0, 1], "i": [1, 2, 1]})
    later = pd.DataFrame({"u": [0, 0, 1], "i": [2, 3, 1]})
    assert new_pairs(later, hist, n_items=10)[["u", "i"]].values.tolist() == [[0, 3]]


def test_time_split_has_no_overlap(project):
    df = load(project["root"] / "events.csv")
    t2 = df.event_time.quantile(0.85)
    before, after = df[df.event_time < t2], df[df.event_time >= t2]
    assert before.event_time.max() < after.event_time.min()
    users, items = kcore(interactions(before))
    hist = to_index(interactions(before), users, items)
    labels = new_pairs(to_index(interactions(after), users, items), hist, len(items))
    assert len(labels.merge(hist, on=["u", "i"])) == 0       # a test label is never in the training history
