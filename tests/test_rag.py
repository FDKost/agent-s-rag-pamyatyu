import os
import pytest
from config import Config
from ingest import ingest_documents
from rag_pipeline import get_chain

@pytest.fixture(scope="module")
def test_config(tmp_path):
    # Create a temporary text file for ingestion
    sample_text = "Hello world. This is a test document. It contains some text for embedding."
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text(sample_text, encoding="utf-8")

    cfg = Config(
        qdrant_collection="test_collection",
        qdrant_vector_size=384,
        qdrant_distance="Cosine",
        in_memory=True,
        ollama_model="llama3",
        embedding_model="all-MiniLM-L6-v2"
    )
    # Ingest the sample document
    ingest_documents(str(tmp_path), cfg)
    return cfg

def test_retrieval_and_response(test_config):
    chain = get_chain(test_config)
    query = "test document"
    response = chain.run(query)
    assert isinstance(response, str)
    assert len(response) > 0
    # Ensure that the response contains some relevant keyword
    assert "test" in response.lower() or "document" in response.lower()
