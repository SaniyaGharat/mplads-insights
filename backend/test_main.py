import asyncio

import pandas as pd
import pytest
from fastapi import HTTPException

from backend.main import LabelRequest, add_label


def test_add_label_rejects_existing_work_index(monkeypatch):
    labels = pd.DataFrame({"label": ["s"], "work_index": [1033]})
    monkeypatch.setattr(pd, "read_csv", lambda _: labels)

    with pytest.raises(HTTPException) as error:
        asyncio.run(add_label(LabelRequest(work_index=1033, label="n")))

    assert error.value.status_code == 409
    assert error.value.detail == "This record has already been labeled"