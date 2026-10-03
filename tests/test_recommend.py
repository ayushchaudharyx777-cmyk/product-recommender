import numpy as np
import pytest


def test_known_user_gets_k_unseen_products_with_reasons(rec):
    uid = int(rec.users[0])
    out = rec.for_user(uid, 10)
    assert len(out) == 10 and out["product_id"].is_unique
    assert not set(out["product_id"]) & set(rec.history(uid)["product_id"])
    assert (out["source"] == "hybrid ranker").all() and out["why"].str.len().gt(0).all()
    assert out["score"].is_monotonic_decreasing


def test_unknown_user_falls_back_to_popularity(rec):
    out = rec.for_user(-1, 5)
    assert len(out) == 5 and out["source"].iloc[0].startswith("popularity")


def test_session_of_known_products_uses_the_ranker(rec):
    picked = [int(p) for p in rec.items[:3]]
    out = rec.for_session(picked, 8)
    assert len(out) == 8 and not set(out["product_id"]) & set(picked)
    assert out["source"].iloc[0] == "hybrid ranker"


def test_new_products_use_the_content_fallback(rec):
    new = rec.all_items[~np.isin(rec.all_items, rec.items)]
    assert len(new) > 0
    pid = int(new[0])
    for out in (rec.similar(pid, 5), rec.for_session([pid], 5)):
        assert len(out) == 5 and pid not in set(out["product_id"])
        assert out["source"].iloc[0].startswith("content fallback")


def test_content_neighbours_share_the_category(rec):
    info = rec.info_all
    pid = int(info[info["category_code"].notna()].index[0])
    out = rec._content([rec.a_pos[pid]], 5, "x")
    assert (out["category_code"] == info.loc[pid, "category_code"]).mean() >= 0.6


def test_unknown_product_raises(rec):
    with pytest.raises(KeyError):
        rec.similar(-5)
    assert rec.for_session([-5], 5)["source"].iloc[0].startswith("popularity")
