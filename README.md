# Personalized Product Recommendation

**Live demo:** https://recommender-ayush.streamlit.app

Hybrid retrieve-and-rank recommender built on the REES46 electronics store event log (views, carts, purchases). Candidates are retrieved from collaborative, sequential and popularity signals, then re-ranked by a LightGBM learning-to-rank model.

## Results

Leave-last-out evaluation on 16,098 users and 3,706 products (top-10 recommendations):

| Model | Hit@10 | NDCG@10 | MRR@10 |
|---|---|---|---|
| Popularity (baseline) | 0.0771 | 0.0423 | 0.0318 |
| SVD (64 factors) | 0.1346 | 0.0799 | 0.0632 |
| Item-item KNN | 0.4392 | 0.2606 | 0.2057 |
| Sequential transitions | 0.4457 | 0.2646 | 0.2088 |
| **Hybrid ranker (LambdaRank)** | **0.5101** | **0.3055** | **0.2425** |

The hybrid ranker puts the user's next product in the top 10 for 51% of users: +16% Hit@10 and +17% NDCG@10 over the best single model, and 6.6x the popularity baseline.

Candidate retrieval returns about 89 products per user with a recall of 0.7875, which is the upper bound for the ranker.

Top ranking features by gain: sequential transition score (30.9%), user category affinity (17.5%), normalised KNN score (11.4%), popularity (9.2%), price ratio to last viewed item (6.7%).

## Dataset

[eCommerce events history in electronics store](https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-electronics-store) (Kaggle, REES46).

- 885,129 events: 793,748 views, 54,035 carts, 37,346 purchases
- 407,283 users, 53,453 products
- `category_code` missing in 26.7% of rows, `brand` in 24.0%

Download `events.csv` and place it at `data/events.csv` (not committed to the repo).

## Approach

**Preprocessing**

- Implicit feedback weights: view = 1, cart = 3, purchase = 5, summed per user-product pair, capped at 20 and log-scaled.
- Iterative k-core filtering: users with at least 3 distinct products, products with at least 5 users. The raw data averages about 2 events per user, so this is what makes collaborative filtering possible.

**Stage 1: candidate retrieval**

- Item-item cosine KNN over the user's full history (top 60)
- Sequential transitions: products opened right after the user's last two products (top 60)
- Most popular products (top 20)

**Stage 2: re-ranking**

LightGBM LambdaRank over 11 features: KNN score (raw and normalised), sequential score, similarity to the last viewed item, popularity, history length, same category, same brand, price, price ratio to the last item, and the share of the user's history in the candidate's category.

**Evaluation without leakage**

The ranker is trained to predict each user's second-last product from the history before it, with retrieval models built only on that earlier history. It is then evaluated on the last product, with retrieval rebuilt on everything except the test items. The baselines use the same test split, so all rows in the table are directly comparable.

## Usage

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python src\train.py
python src\hybrid_ranker.py
python src\recommend.py --user 1515915625353230683
python src\recommend.py --item 1821813
```

`--user` returns personalised recommendations from the hybrid ranker, `--item` returns similar products. Users not in the model get the most popular products.

## Project structure

```
product-recommender/
├── data/events.csv             # not committed
├── models/metrics.json         # baseline results
├── models/metrics_ranker.json  # hybrid ranker results
├── src/train.py                # preprocessing, baselines (popularity, KNN, SVD)
├── src/hybrid_ranker.py        # candidate retrieval + LambdaRank re-ranker
├── src/recommend.py            # recommendation CLI
└── requirements.txt
```

## Limitations

- Only 16,098 of 407,283 users (about 4%) pass the filter. Everyone else gets the popularity fallback.
- Products with fewer than 5 users are not recommendable.
- 21% of test products are never retrieved in stage 1, so the ranker cannot recover them.
- Leave-last-out lets retrieval models use other users' later events, so a strict time-based split would score lower.
- Offline metrics only. No online or A/B evaluation.

## Future Scope

- Better retrieval to lift the 0.79 recall ceiling: ALS or two-tower embeddings with approximate nearest neighbour search
- Session-based models (GRU4Rec, SASRec) to model the full order of events
- Content-based fallback using category, brand and price for cold-start users and products
- Global time-based split and hyperparameter search for the ranker
- REST API (FastAPI) with Docker, plus monitoring of coverage and drift