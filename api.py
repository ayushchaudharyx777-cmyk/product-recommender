"""Recommendation API.  Run:  uvicorn api:app --reload   (docs at http://localhost:8000/docs)"""
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent / "src"))
from recommend import Recommender  # noqa: E402

state = {}


@asynccontextmanager
async def lifespan(app):
    state["rec"] = Recommender(os.environ.get("MODEL_PATH", "models/recommender.joblib"))
    yield


app = FastAPI(title="Product Recommendation API", version="1.0", lifespan=lifespan)


class Session(BaseModel):
    product_ids: list[int] = Field(..., min_length=1, description="Viewed products, oldest first")
    k: int = Field(10, ge=1, le=50)


def records(df):
    return json.loads(df.to_json(orient="records"))      # NaN -> null


@app.get("/health")
def health():
    rec = state["rec"]
    return {"status": "ok", "users_in_model": len(rec.users), "products_in_model": len(rec.items),
            "catalogue_products": len(rec.all_items)}


@app.get("/recommend/user/{user_id}")
def recommend_user(user_id: int, k: int = Query(10, ge=1, le=50)):
    """Personalised recommendations; users the model has never seen get popular products."""
    rec = state["rec"]
    return {"user_id": user_id, "known_user": user_id in rec.u_pos, "items": records(rec.for_user(user_id, k))}


@app.post("/recommend/session")
def recommend_session(s: Session):
    """Recommendations from a list of viewed products. Products outside the collaborative model use the
    content-based fallback."""
    return {"items": records(state["rec"].for_session(s.product_ids, s.k))}


@app.get("/similar/{product_id}")
def similar(product_id: int, k: int = Query(10, ge=1, le=50)):
    try:
        return {"product_id": product_id, "items": records(state["rec"].similar(product_id, k))}
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


@app.get("/metrics")
def metrics():
    path = Path(os.environ.get("REPORT_DIR", "reports")) / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "run src/train.py first")
    return json.load(open(path))
