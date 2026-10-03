# Results

Data: 884,474 events, 407,283 users, 53,453 products (2020-09-24 to 2021-02-28).

## 1. Time-based split

Everything before 2021-01-30 is training data; the test is the new products each returning user interacts with afterwards. 317 test users, 2.94 relevant products per user on average. The ranker is trained on next-product labels of 9,760 users, all taken from before the cut-off.

| Model | precision@5 | recall@5 | ndcg@5 | precision@10 | recall@10 | ndcg@10 | precision@20 | recall@20 | ndcg@20 |
|---|---|---|---|---|---|---|---|---|---|
| Matrix factorisation (SVD) | 0.0461 | 0.0909 | 0.0851 | 0.0341 | 0.1275 | 0.0945 | 0.0263 | 0.1996 | 0.1155 |
| Popularity | 0.0568 | 0.1241 | 0.1159 | 0.0413 | 0.1834 | 0.1345 | 0.0274 | 0.2369 | 0.1492 |
| Content-based | 0.0669 | 0.1725 | 0.1367 | 0.0467 | 0.2321 | 0.1557 | 0.0358 | 0.3336 | 0.1859 |
| Sequential transitions | 0.0738 | 0.1815 | 0.1405 | 0.0574 | 0.2590 | 0.1674 | 0.0402 | 0.3593 | 0.1967 |
| Item-item KNN | 0.0751 | 0.1887 | 0.1503 | 0.0590 | 0.2772 | 0.1798 | 0.0457 | 0.3915 | 0.2153 |
| Hybrid ranker (LambdaRank) | 0.0883 | 0.2122 | 0.1687 | 0.0653 | 0.2969 | 0.1962 | 0.0514 | 0.4461 | 0.2414 |

Candidate retrieval: 114.6 products per user, recall 0.6352 (upper bound for the ranker).

Ranker minus item-KNN on NDCG@10: +0.0164 (95% bootstrap CI -0.0061 to +0.0374, not significant: interval includes 0).

### Beyond accuracy (top 10)

| Model | coverage | diversity | novelty |
|---|---|---|---|
| Popularity | 0.0059 | 0.3966 | 6.5800 |
| Matrix factorisation (SVD) | 0.1334 | 0.3298 | 9.5000 |
| Item-item KNN | 0.3641 | 0.1850 | 10.0400 |
| Sequential transitions | 0.3537 | 0.2285 | 9.6800 |
| Content-based | 0.3363 | 0.0142 | 10.1400 |
| Hybrid ranker (LambdaRank) | 0.3443 | 0.1195 | 9.5800 |

Coverage = share of the catalogue that is ever recommended. Diversity = share of product pairs in a list that come from different categories. Novelty = mean self-information of recommended products (higher = less popular).

## 2. Leave-last-out (next-product prediction)

16,098 users; the last product of each user is hidden, so Recall@K equals Hit@K.

| Model | precision@5 | recall@5 | ndcg@5 | precision@10 | recall@10 | ndcg@10 | precision@20 | recall@20 | ndcg@20 |
|---|---|---|---|---|---|---|---|---|---|
| Popularity | 0.0099 | 0.0494 | 0.0333 | 0.0077 | 0.0771 | 0.0423 | 0.0062 | 0.1249 | 0.0545 |
| Matrix factorisation (SVD) | 0.0190 | 0.0952 | 0.0673 | 0.0135 | 0.1346 | 0.0799 | 0.0093 | 0.1859 | 0.0928 |
| Content-based | 0.0460 | 0.2300 | 0.1579 | 0.0324 | 0.3239 | 0.1883 | 0.0216 | 0.4312 | 0.2152 |
| Item-item KNN | 0.0643 | 0.3217 | 0.2226 | 0.0439 | 0.4392 | 0.2607 | 0.0278 | 0.5566 | 0.2903 |
| Sequential transitions | 0.0649 | 0.3245 | 0.2242 | 0.0437 | 0.4370 | 0.2607 | 0.0268 | 0.5363 | 0.2859 |
| Hybrid ranker (LambdaRank) | 0.0767 | 0.3833 | 0.2658 | 0.0522 | 0.5219 | 0.3106 | 0.0325 | 0.6495 | 0.3430 |

Candidate recall 0.8445.

Ranker minus item-KNN on NDCG@10: +0.0500 (95% bootstrap CI +0.0461 to +0.0540, significant).

### Ablation: retrain the ranker without one feature group (leave-last-out)

| Ranker | ndcg@10 | recall@10 |
|---|---|---|
| all features | 0.3106 | 0.5219 |
| without collaborative | 0.2947 | 0.4996 |
| without sequential | 0.3021 | 0.5098 |
| without content | 0.2965 | 0.4975 |
| without popularity | 0.3073 | 0.5156 |

### User segments (hybrid ranker, leave-last-out)

| Segment type | segment | users | precision@10 | recall@10 | ndcg@10 | knn_ndcg@10 | lift_vs_knn_pct |
|---|---|---|---|---|---|---|---|
| Favourite category | other categories | 5862 | 0.0547 | 0.5466 | 0.3277 | 0.2692 | 21.7490 |
| Favourite category | videocards | 4957 | 0.0455 | 0.4545 | 0.2571 | 0.2212 | 16.2349 |
| Favourite category | unknown | 2461 | 0.0577 | 0.5766 | 0.3467 | 0.2904 | 19.4111 |
| Favourite category | motherboard | 1037 | 0.0618 | 0.6181 | 0.4200 | 0.3966 | 5.9214 |
| Favourite category | telephone | 1009 | 0.0417 | 0.4172 | 0.2458 | 0.1919 | 28.0571 |
| Favourite category | tv | 772 | 0.0601 | 0.6010 | 0.3469 | 0.2618 | 32.4848 |
| History length | 1-2 products | 7215 | 0.0587 | 0.5871 | 0.3545 | 0.3021 | 17.3474 |
| History length | 3-4 products | 5009 | 0.0528 | 0.5277 | 0.3108 | 0.2620 | 18.6622 |
| History length | 5-9 products | 2803 | 0.0412 | 0.4124 | 0.2388 | 0.1938 | 23.2029 |
| History length | 10+ products | 1071 | 0.0343 | 0.3427 | 0.2017 | 0.1501 | 34.3248 |
| Price tier | budget | 5367 | 0.0602 | 0.6020 | 0.3746 | 0.3249 | 15.2933 |
| Price tier | premium | 5366 | 0.0480 | 0.4799 | 0.2770 | 0.2381 | 16.3458 |
| Price tier | mid-range | 5365 | 0.0484 | 0.4839 | 0.2802 | 0.2189 | 27.9875 |
| Shopper type | browser only | 13102 | 0.0535 | 0.5349 | 0.3175 | 0.2664 | 19.1751 |
| Shopper type | has purchased | 2996 | 0.0465 | 0.4653 | 0.2805 | 0.2355 | 19.1288 |

## 3. Cold start: content-based fallback

5,000 users outside the collaborative model; their last product is predicted from the earlier ones using metadata only, ranked against all 53,453 catalogue products. 56.0% of the hidden products are not in the collaborative model at all.

| Model | hit@10 | ndcg@10 |
|---|---|---|
| Popularity | 0.0334 | 0.0166 |
| Content-based fallback | 0.2858 | 0.1711 |
