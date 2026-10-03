# Personalized Product Recommendation

Implicit-feedback recommender built on the REES46 electronics store event log (views, carts, purchases). Compares three models and serves the best one through a small CLI.

## Results

Leave-last-out evaluation on 16,098 users and 3,706 products (top-10 recommendations):

| Model | Hit@10 | NDCG@10 | MRR@10 | Coverage |
|---|---|---|---|---|
| Popularity (baseline) | 0.0771 | 0.0423 | 0.0318 | 0.0081 |
| **Item-item KNN** | **0.4392** | **0.2606** | **0.2057** | **0.9873** |
| SVD (64 factors) | 0.1346 | 0.0799 | 0.0632 | 0.1341 |

Item-KNN puts the user's next product in the top 10 for 44% of users, about 5.7x the popularity baseline, while recommending from 98.7% of the catalogue.

## Dataset

[eCommerce events history in electronics store](https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-electronics-store) (Kaggle, REES46).

- 885,129 events: 793,748 views, 54,035 carts, 37,346 purchases
- 407,283 users, 53,453 products
- `category_code` missing in 26.7% of rows, `brand` in 24.0%

Download `events.csv` and place it at `data/events.csv` (not committed to the repo).

## Approach

1. **Implicit feedback weights**: view = 1, cart = 3, purchase = 5. Weights are summed per user-product pair, capped at 20, then log-scaled.
2. **k-core filtering**: keep users with at least 3 distinct products and products with at least 5 users, applied iteratively. The raw data averages about 2 events per user, so this step is what makes collaborative filtering possible.
3. **Split**: leave-last-out. Each user's most recently discovered product is held out as the test item.
4. **Models**: popularity baseline, item-item cosine KNN (top 50 neighbours per item), truncated SVD.
5. **Serving**: the best model is refit on all interactions and saved with product metadata.

## Usage

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python src\train.py
python src\recommend.py --item 1821813
python src\recommend.py --user 1515915625353230683
```

`--item` returns similar products, `--user` returns personalised recommendations. Users not in the model get the most popular products.

## Project structure

```
product-recommender/
├── data/events.csv          # not committed
├── models/metrics.json      # evaluation results
├── src/train.py             # preprocessing, training, evaluation
├── src/recommend.py         # recommendation CLI
└── requirements.txt
```

## Limitations

- Only 16,098 of 407,283 users (about 4%) pass the filter. Everyone else gets the popularity fallback.
- Products with fewer than 5 users are not recommendable.
- Leave-last-out lets item similarities use other users' later events, so a strict time-based split would score lower.
- Offline metrics only. No online or A/B evaluation.

## Future Scope

- Content-based fallback using category, brand and price for cold-start users and products
- Session-based models (GRU4Rec, SASRec) to use the order of events within a session
- ALS or BPR tuned for implicit feedback
- Global time-based split and hyperparameter search
- REST API (FastAPI) with Docker, plus monitoring of recommendation coverage and drift
