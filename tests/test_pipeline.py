import json

import numpy as np

from retrieval import FEATURES, content_matrix

RANKER = "Hybrid ranker (LambdaRank)"


def test_artifacts_and_reports_exist(project):
    root = project["root"]
    for f in ["models/recommender.joblib", "reports/metrics.json", "reports/RESULTS.md", "reports/segments.csv",
              "reports/model_comparison.png", "reports/features_ablation.png", "reports/segments.png"]:
        assert (root / f).exists(), f
    assert json.load(open(root / "reports/metrics.json"))["time_split"]["n_test_users"] > 0


def test_every_model_reports_valid_metrics(project):
    for split in ("time_split", "leave_last_out"):
        block = project["metrics"][split]
        assert 0 < block["candidate_recall"] <= 1
        for name, m in block["models"].items():
            for k in (5, 10, 20):
                for metric in ("precision", "recall", "ndcg"):
                    assert 0 <= m[f"{metric}@{k}"] <= 1, (split, name)
            assert m["recall@5"] <= m["recall@10"] <= m["recall@20"]


def test_ranker_beats_popularity_baseline(project):
    for split in ("time_split", "leave_last_out"):
        m = project["metrics"][split]["models"]
        assert m[RANKER]["ndcg@10"] > m["Popularity"]["ndcg@10"]


def test_ranker_cannot_exceed_candidate_recall(project):
    block = project["metrics"]["leave_last_out"]
    assert block["models"][RANKER]["recall@20"] <= block["candidate_recall"] + 1e-9


def test_ablation_covers_every_feature_group(project):
    ab = project["metrics"]["leave_last_out"]["ablation"]
    assert set(ab) == {"all features", "without collaborative", "without sequential", "without content",
                       "without popularity"}
    assert set(project["metrics"]["leave_last_out"]["feature_importance_pct"]) == set(FEATURES)


def test_content_vectors_need_no_interactions(rec):
    V = content_matrix(rec.info_all.assign(category_id=np.arange(len(rec.info_all)) % 7))
    assert V.shape[0] == len(rec.info_all)
    assert np.allclose(np.asarray(V.multiply(V).sum(1)).ravel(), 1, atol=1e-5)   # unit length


def test_content_fallback_beats_popularity_for_cold_users(project):
    m = project["metrics"]["cold_start"]["models"]
    assert m["Content-based fallback"]["hit@10"] > m["Popularity"]["hit@10"]
