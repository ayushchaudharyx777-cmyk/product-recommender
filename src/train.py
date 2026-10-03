"""End-to-end pipeline: data -> retrieval -> LambdaRank re-ranker -> evaluation -> serving artifacts.

Run from the project root:  python src/train.py            (add --fast for a quick run)
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np

from data import interactions, item_meta, kcore, label_matrix, load, new_pairs, to_index
from evaluate import cold_start_eval, run_protocol, segment_table
from reports import write_reports
from retrieval import ItemFeatures, Retriever, content_matrix

TEST_START = 0.80      # time quantile of the event log: everything after it is the test period


def main(data_path="data/events.csv", model_dir="models", report_dir="reports", fast=False, log=print):
    t0 = time.time()
    model_dir, report_dir = Path(model_dir), Path(report_dir)
    model_dir.mkdir(exist_ok=True); report_dir.mkdir(exist_ok=True)
    df = load(data_path)
    meta_all = item_meta(df)
    ui = interactions(df)
    metrics = {"data": {"events": int(len(df)), "users": int(df.user_id.nunique()),
                        "products": int(len(meta_all)), "first_event": str(df.event_time.min().date()),
                        "last_event": str(df.event_time.max().date())}}

    # 1) Time-based split (the protocol the brief asks for)
    t_test = df.event_time.quantile(TEST_START)
    log(f"[1/4] time-based split: train before {t_test.date()}, test on new products after it")
    ui_b = interactions(df[df.event_time < t_test])
    users, items = kcore(ui_b)
    shape = (len(users), len(items))
    feats = ItemFeatures(meta_all.reindex(items))
    hist_b = to_index(ui_b, users, items)
    lab_b = new_pairs(to_index(interactions(df[df.event_time >= t_test]), users, items), hist_b, shape[1])
    # The ranker learns from next-product labels taken entirely from BEFORE the cut-off:
    # history = everything but each user's last pre-cut-off product, label = that product.
    back = hist_b.groupby("u").cumcount(ascending=False).to_numpy()
    res_time = run_protocol(hist_b[back >= 1], label_matrix(hist_b[back == 0], shape),
                            hist_b, label_matrix(lab_b, shape), shape, feats, fast=fast, log=log)
    metrics["time_split"] = {"test_start": str(t_test.date()), "users": shape[0], "products": shape[1],
                             **res_time.summary}

    # 2) Leave-last-out (next-product prediction) on the full universe; large enough for ablation and segments
    log("[2/4] leave-last-out split")
    users, items = kcore(ui)
    shape = (len(users), len(items))
    feats = ItemFeatures(meta_all.reindex(items))
    hist = to_index(ui, users, items)
    back = hist.groupby("u").cumcount(ascending=False).to_numpy()   # 0 = last product, 1 = second last
    res_llo = run_protocol(hist[back >= 2], label_matrix(hist[back == 1], shape),
                           hist[back >= 1], label_matrix(hist[back == 0], shape), shape, feats,
                           fast=fast, ablation=True, log=log)
    metrics["leave_last_out"] = {"users": shape[0], "products": shape[1], **res_llo.summary}
    segments = segment_table(hist[back >= 1], res_llo, feats)

    # 3) Content-based fallback for users and products outside the collaborative model
    log("[3/4] cold-start evaluation of the content-based fallback")
    metrics["cold_start"] = cold_start_eval(ui, users, items, meta_all, n_sample=1000 if fast else 5000)

    # 4) Serving artifacts: retrieval refitted on every interaction, plus the full-catalogue content index
    log("[4/4] saving serving artifacts and reports")
    all_items = meta_all.index.to_numpy()
    pop_all = np.bincount(meta_all.index.get_indexer(ui.product_id), minlength=len(all_items)).astype(float)
    metrics["serving"] = {"users_in_model": shape[0], "products_in_model": shape[1],
                          "catalogue_products": int(len(all_items))}
    joblib.dump({"ranker": res_llo.ranker, "ret": Retriever(hist, shape), "feats": feats, "users": users,
                 "items": items, "all_items": all_items, "V_all": content_matrix(meta_all),
                 "info_all": meta_all[["category_code", "brand", "price"]],
                 "popn_all": np.log1p(pop_all) / np.log1p(pop_all).max()},
                model_dir / "recommender.joblib", compress=3)
    json.dump(metrics, open(report_dir / "metrics.json", "w"), indent=2)
    segments.to_csv(report_dir / "segments.csv", index=False)
    write_reports(metrics, segments, report_dir)
    log(f"done in {time.time() - t0:.0f}s -> {model_dir}/recommender.joblib, {report_dir}/RESULTS.md")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/events.csv")
    ap.add_argument("--fast", action="store_true", help="smaller ranker and cold-start sample")
    args = ap.parse_args()
    m = main(args.data, fast=args.fast)
    print(open("reports/RESULTS.md", encoding="utf-8").read())
