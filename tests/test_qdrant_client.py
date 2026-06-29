import pytest
from qdrant_client import QdrantWrapper

@pytest.mark.skipif(
    not pytest.config.getoption("--qdrant"),
    reason="Qdrant server not available"
)
def test_upsert_and_search():
    wrapper = QdrantWrapper()
    text = "Test document for Qdrant"
    embedding = [0.1] * 384
    metadata = {"source": "unit_test"}
    point_id = wrapper.upsert(text, embedding, metadata)
    assert point_id is not None
    results = wrapper.search(embedding, limit=1)
    assert len(results) > 0
    assert results[0].payload.get("text") == text
