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
        "work_index", "global_rank", "MP", "IDA", "amount", "sanction_lag_days",
        "status", "category", "score", "label_status"
    }
    assert response["data"][0]["global_rank"] == 1
    assert response["data"][1]["global_rank"] == 2


def test_risk_scores_search_raghavendra():
    import backend.main as bm
    import numpy as np

    if "df_works" not in bm.DATA:
        bm.DATA["df_works"] = pd.read_csv("data/works_sanctioned_clean.csv")
        bm.DATA["labels_df"] = pd.read_csv("ml/anomaly_labels.csv")
        bm.DATA["gat_scores"] = np.load("ml/gat_scores.npy")

    # Tests case-insensitivity, whitespace collapsing, and ignoring (2024-2029)
    response = asyncio.run(bm.get_risk_scores(search="raghavendra"))
    assert response["total"] == 100
    assert len(response["data"]) == 50
    assert "Raghavendra" in response["data"][0]["MP"]
    assert response["data"][0]["global_rank"] == 22

    # Query with irregular spaces
    response_spaced = asyncio.run(bm.get_risk_scores(search="  b   y   raghavendra  "))
    assert response_spaced["total"] == 100
    assert response_spaced["data"][0]["global_rank"] == 22


def test_risk_scores_state_district_filter_and_global_rank():
    import backend.main as bm
    import numpy as np

    if "df_works" not in bm.DATA:
        bm.DATA["df_works"] = pd.read_csv("data/works_sanctioned_clean.csv")
        bm.DATA["labels_df"] = pd.read_csv("ml/anomaly_labels.csv")
        bm.DATA["gat_scores"] = np.load("ml/gat_scores.npy")

    response = asyncio.run(bm.get_risk_scores(state="Karnataka", district="DHARWAD"))
    assert response["total"] == 15
    assert len(response["data"]) == 15
    for row in response["data"]:
        assert row["IDA"].startswith("DHARWAD")
    # First row is work_index 1 with global_rank 38; rank is NOT reset to 1
    assert response["data"][0]["work_index"] == 1
    assert response["data"][0]["global_rank"] == 38


def test_filter_options_endpoint():
    import backend.main as bm
    import numpy as np

    if "df_works" not in bm.DATA:
        bm.DATA["df_works"] = pd.read_csv("data/works_sanctioned_clean.csv")
        bm.DATA["labels_df"] = pd.read_csv("ml/anomaly_labels.csv")
        bm.DATA["gat_scores"] = np.load("ml/gat_scores.npy")

    options = asyncio.run(bm.get_filter_options())
    assert "states" in options
    assert "Karnataka" in options["states"]
    assert "districts_by_state" in options
    assert "DHARWAD" in options["districts_by_state"]["Karnataka"]
    assert "statuses" in options
    assert "Work Completed" in options["statuses"]
    assert options["amount_min"] == 12881.0
    assert options["lag_min"] == 0
    assert options["lag_max"] == 732