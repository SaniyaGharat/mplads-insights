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
        "status", "category", "score", "label_status", "review_flag"
    }
    assert response["data"][0]["global_rank"] == 1
    assert response["data"][1]["global_rank"] == 2
    assert response["data"][0]["review_flag"] is True


def test_risk_scores_flag_pct():
    import backend.main as bm
    import numpy as np

    if "df_works" not in bm.DATA:
        bm.DATA["df_works"] = pd.read_csv("data/works_sanctioned_clean.csv")
        bm.DATA["labels_df"] = pd.read_csv("ml/anomaly_labels.csv")
        bm.DATA["gat_scores"] = np.load("ml/gat_scores.npy")

    res_1 = asyncio.run(bm.get_risk_scores(flag_pct=1))
    assert res_1["data"][0]["review_flag"] is True

    # At page 4, index 151 (rank 151 > 150 for 1%), review_flag is False
    res_page4 = asyncio.run(bm.get_risk_scores(page=4, page_size=50, flag_pct=1))
    assert res_page4["data"][1]["global_rank"] == 152
    assert res_page4["data"][1]["review_flag"] is False


def test_work_compare_endpoint():
    import backend.main as bm
    import numpy as np

    if "df_works" not in bm.DATA:
        bm.DATA["df_works"] = pd.read_csv("data/works_sanctioned_clean.csv")
        bm.DATA["labels_df"] = pd.read_csv("ml/anomaly_labels.csv")
        bm.DATA["gat_scores"] = np.load("ml/gat_scores.npy")

    res = asyncio.run(bm.compare_work(0))
    assert res["work_index"] == 0
    assert res["total_works"] == 15000
    assert set(res["methods"].keys()) == {"gat", "siamese", "community", "fusion"}
    for m in ["gat", "siamese", "community", "fusion"]:
        m_info = res["methods"][m]
        assert "global_rank" in m_info
        assert "percentile" in m_info
        assert "score" in m_info
        assert "direction" in m_info
        assert 1 <= m_info["global_rank"] <= 15000


def test_evaluation_endpoint():
    import backend.main as bm

    res = asyncio.run(bm.get_evaluation())
    assert "review_budget_recall" in res
    assert "p1_pooled_summary" in res
    assert "p2_bootstrap_primary_fgh" in res
    assert "limitations" in res
    assert len(res["review_budget_recall"]) > 0
    assert len(res["p1_pooled_summary"]) > 0
    assert len(res["p2_bootstrap_primary_fgh"]) > 0
    assert len(res["limitations"]) >= 3


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
