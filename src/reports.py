"""Write reports/RESULTS.md and the charts used in the README and the dashboard."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RANKER, KNN = "Hybrid ranker (LambdaRank)", "Item-item KNN"
COLS = ["precision@5", "recall@5", "ndcg@5", "precision@10", "recall@10", "ndcg@10",
        "precision@20", "recall@20", "ndcg@20"]


def md_table(df, floatfmt="{:.4f}"):
    head = "| " + " | ".join([df.index.name or ""] + [str(c) for c in df.columns]) + " |"
    lines = [head, "|" + "---|" * (len(df.columns) + 1)]
    for idx, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def model_table(block):
    df = pd.DataFrame(block["models"]).T[COLS].sort_values("ndcg@10")
    df.index.name = "Model"
    return df


def ci_text(ci):
    return (f"Ranker minus item-KNN on NDCG@10: {ci['diff']:+.4f} (95% bootstrap CI {ci['ci_low']:+.4f} to "
            f"{ci['ci_high']:+.4f}, {'significant' if ci['significant'] else 'not significant: interval includes 0'}).")


def write_reports(metrics, segments, out):
    ts, llo, cold = metrics["time_split"], metrics["leave_last_out"], metrics["cold_start"]
    ci = ts["ranker_vs_knn_ndcg@10"]
    seg = segments.rename(columns={"segment_type": "Segment type"}).set_index("Segment type")
    parts = [
        "# Results", "",
        f"Data: {metrics['data']['events']:,} events, {metrics['data']['users']:,} users, "
        f"{metrics['data']['products']:,} products ({metrics['data']['first_event']} to "
        f"{metrics['data']['last_event']}).", "",
        "## 1. Time-based split", "",
        f"Everything before {ts['test_start']} is training data; the test is the new products each returning "
        f"user interacts with afterwards. {ts['n_test_users']:,} test users, "
        f"{ts['relevant_items_per_test_user']} relevant products per user on average. The ranker is trained on "
        f"next-product labels of {ts['n_train_users']:,} users, all taken from before the cut-off.", "",
        md_table(model_table(ts)), "",
        f"Candidate retrieval: {ts['candidates_per_user']} products per user, recall {ts['candidate_recall']} "
        "(upper bound for the ranker).", "",
        ci_text(ci), "",
        "### Beyond accuracy (top 10)", "",
        md_table(pd.DataFrame(ts["beyond_accuracy"]).T.rename_axis("Model")), "",
        "Coverage = share of the catalogue that is ever recommended. Diversity = share of product pairs in a list "
        "that come from different categories. Novelty = mean self-information of recommended products (higher = "
        "less popular).", "",
        "## 2. Leave-last-out (next-product prediction)", "",
        f"{llo['n_test_users']:,} users; the last product of each user is hidden, so Recall@K equals Hit@K.", "",
        md_table(model_table(llo)), "",
        f"Candidate recall {llo['candidate_recall']}.", "", ci_text(llo["ranker_vs_knn_ndcg@10"]), "",
        "### Ablation: retrain the ranker without one feature group (leave-last-out)", "",
        md_table(pd.DataFrame(llo["ablation"]).T.rename_axis("Ranker")), "",
        "### User segments (hybrid ranker, leave-last-out)", "",
        md_table(seg), "",
        "## 3. Cold start: content-based fallback", "",
        f"{cold['n_users']:,} users outside the collaborative model; their last product is predicted from the earlier "
        f"ones using metadata only, ranked against all {cold['catalogue_size']:,} catalogue products. "
        f"{cold['targets_outside_cf_model_pct']}% of the hidden products are not in the collaborative model at all.",
        "", md_table(pd.DataFrame(cold["models"]).T.rename_axis("Model")), "",
    ]
    (out / "RESULTS.md").write_text("\n".join(parts), encoding="utf-8")

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    df = model_table(ts)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    colors = ["#1f77b4" if m != RANKER else "#d95f02" for m in df.index]
    ax[0].barh(df.index, df["ndcg@10"], color=colors)
    ax[0].set_title("NDCG@10, time-based split")
    for k, style in (("precision", "--"), ("recall", "-")):
        for m, col in ((RANKER, "#d95f02"), (KNN, "#1f77b4"), ("Popularity", "#7f7f7f")):
            ax[1].plot([5, 10, 20], [ts["models"][m][f"{k}@{n}"] for n in (5, 10, 20)], style, color=col,
                       marker="o", label=f"{m} {k}")
    ax[1].set_xticks([5, 10, 20]); ax[1].set_xlabel("K"); ax[1].set_title("Precision@K and Recall@K")
    ax[1].legend(fontsize=6.5, frameon=False)
    fig.tight_layout(); fig.savefig(out / "model_comparison.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    imp = pd.Series(llo["feature_importance_pct"]).sort_values()
    ax[0].barh(imp.index, imp.values, color="#1f77b4"); ax[0].set_title("Ranker feature importance (gain %)")
    ab = pd.DataFrame(llo["ablation"]).T["ndcg@10"]
    ax[1].barh(ab.index[::-1], ab.values[::-1], color=["#1f77b4"] * (len(ab) - 1) + ["#d95f02"])
    ax[1].set_title("Ablation: NDCG@10 without a feature group (leave-last-out)")
    fig.tight_layout(); fig.savefig(out / "features_ablation.png", dpi=140); plt.close(fig)

    s = segments[segments.segment_type.isin(["History length", "Shopper type", "Price tier"])]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    x = range(len(s))
    ax.bar([i - 0.2 for i in x], s["knn_ndcg@10"], 0.4, label=KNN, color="#1f77b4")
    ax.bar([i + 0.2 for i in x], s["ndcg@10"], 0.4, label=RANKER, color="#d95f02")
    ax.set_xticks(list(x)); ax.set_xticklabels([f"{a}\n(n={n})" for a, n in zip(s.segment, s.users)], fontsize=7)
    ax.set_title("NDCG@10 by user segment (leave-last-out)"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "segments.png", dpi=140); plt.close(fig)
