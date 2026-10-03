import numpy as np

from metrics import beyond_accuracy, top_k_dense, top_k_pairs, user_metrics


def test_precision_recall_ndcg_by_hand():
    hits = np.zeros((1, 20), dtype=bool)
    hits[0, [0, 2]] = True                                   # relevant items at ranks 1 and 3, 4 relevant in total
    m = user_metrics(hits, np.array([4])).iloc[0]
    assert np.isclose(m["precision@5"], 2 / 5) and np.isclose(m["recall@5"], 2 / 4)
    dcg = 1 / np.log2(2) + 1 / np.log2(4)
    idcg = sum(1 / np.log2(i + 2) for i in range(4))
    assert np.isclose(m["ndcg@5"], dcg / idcg) and m["hit@5"] == 1


def test_perfect_ranking_scores_one():
    hits = np.zeros((1, 20), dtype=bool)
    hits[0, :3] = True
    m = user_metrics(hits, np.array([3])).iloc[0]
    assert np.isclose(m["ndcg@10"], 1) and np.isclose(m["recall@5"], 1)


def test_top_k_never_returns_seen_items():
    sc = np.array([[5.0, 4.0, 3.0, 2.0]])
    seen = np.array([[True, False, False, False]])
    assert top_k_dense(sc, seen, 2).tolist() == [[1, 2]]
    assert top_k_dense(sc, np.array([[True, True, True, False]]), 2).tolist() == [[3, -1]]


def test_top_k_pairs_orders_by_score_within_user():
    top = top_k_pairs(np.array([0.1, 0.9, 0.5, 0.3]), np.array([0, 0, 0, 1]), np.array([7, 8, 9, 4]), 2, k=2)
    assert top.tolist() == [[8, 9], [4, -1]]


def test_diversity_is_zero_for_single_category_lists():
    topk = np.arange(10)[None, :]
    out = beyond_accuracy(topk, cat=np.zeros(10, dtype=int), pop=np.ones(10), k=10)
    assert out["diversity"] == 0 and out["coverage"] == 1
