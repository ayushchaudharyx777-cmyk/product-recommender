import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(project, rec):
    import os
    os.environ["MODEL_PATH"] = str(project["root"] / "models" / "recommender.joblib")
    os.environ["REPORT_DIR"] = str(project["root"] / "reports")
    import api
    with TestClient(api.app) as c:
        yield c


def test_health(client, rec):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["products_in_model"] == len(rec.items)


def test_recommend_user(client, rec):
    r = client.get(f"/recommend/user/{int(rec.users[0])}", params={"k": 7})
    assert r.status_code == 200 and r.json()["known_user"] and len(r.json()["items"]) == 7
    assert not client.get("/recommend/user/1").json()["known_user"]


def test_recommend_session(client, rec):
    r = client.post("/recommend/session", json={"product_ids": [int(p) for p in rec.items[:2]], "k": 5})
    assert r.status_code == 200 and len(r.json()["items"]) == 5
    assert client.post("/recommend/session", json={"product_ids": []}).status_code == 422


def test_similar_and_404(client, rec):
    assert len(client.get(f"/similar/{int(rec.items[0])}").json()["items"]) == 10
    assert client.get("/similar/-1").status_code == 404
    assert client.get("/recommend/user/1", params={"k": 500}).status_code == 422


def test_metrics_endpoint(client):
    assert "time_split" in client.get("/metrics").json()
