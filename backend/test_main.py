import asyncio

import pandas as pd
import pytest
from fastapi import HTTPException

from backend.main import LabelRequest, add_label


def test_add_label_rejects_existing_work_index(monkeypatch):
    monkeypatch.setenv("LABELLING_ENABLED", "true")
    labels = pd.DataFrame({"label": ["s"], "work_index": [1033]})
    monkeypatch.setattr(pd, "read_csv", lambda _: labels)

    with pytest.raises(HTTPException) as error:
        asyncio.run(add_label(LabelRequest(work_index=1033, label="n")))

    assert error.value.status_code == 409
    assert error.value.detail == "This record has already been labeled"


def test_add_label_disabled_when_flag_false(monkeypatch):
    monkeypatch.delenv("LABELLING_ENABLED", raising=False)
    with pytest.raises(HTTPException) as error:
        asyncio.run(add_label(LabelRequest(work_index=1033, label="n")))

    assert error.value.status_code == 403
    assert error.value.detail == "Labelling is disabled in this deployment"


def test_risk_scores_unfiltered_defaults():
    import backend.main as bm
    import numpy as np

    if "df_works" not in bm.DATA:
        bm.DATA["df_works"] = pd.read_csv("data/works_sanctioned_clean.csv")
        bm.DATA["labels_df"] = pd.read_csv("ml/anomaly_labels.csv")
        bm.DATA["gat_scores"] = np.load("ml/gat_scores.npy")

    response = asyncio.run(bm.get_risk_scores())
    assert set(response.keys()) == {"data", "page", "page_size", "total", "method", "method_note"}
    assert response["total"] == 15000
    assert response["page"] == 1
    assert response["page_size"] == 50
    assert response["method"] == "gat"
    assert (
        response["method_note"]
        == "Reconstruction error: flags statistically unusual works. Favours rare values; not evidence of misconduct."
    )
    assert len(response["data"]) == 50
    assert set(response["data"][0].keys()) == {
        "work_index", "MP", "IDA", "amount", "sanction_lag_days",
        "status", "category", "score", "label_status"
    }