"""Shared fixtures: a small SYNTHETIC event log and one full pipeline run for the whole test session."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def project(tmp_path_factory):
    import train
    from make_synthetic_data import make
    root = tmp_path_factory.mktemp("proj")
    make(root / "events.csv", n_users=2500, n_items=600, n_cats=10, seed=1)
    metrics = train.main(root / "events.csv", root / "models", root / "reports", fast=True, log=lambda *a: None)
    return {"root": root, "metrics": metrics}


@pytest.fixture(scope="session")
def rec(project):
    from recommend import Recommender
    return Recommender(project["root"] / "models" / "recommender.joblib")
