import os
import pytest
from agent import load_and_index_documents, query_agent, get_qdrant_client, COLLECTION_NAME

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Ensure Qdrant is running
    client = get_qdrant_client()
    # Clean up any existing collection
    if COLLECTION_NAME in client.get_collections().collections:
        client.delete_collection(COLLECTION_NAME)
    # Index documents
    load_and_index_documents()
    yield
    # Teardown: delete collection
    client.delete_collection(COLLECTION_NAME)

def test_query_returns_answer():
    question = "What is a pangram?"
    answer, sources = query_agent(question)
    assert isinstance(answer, str)
    assert len(answer) > 0
    assert isinstance(sources, str)
    assert len(sources) > 0

def test_query_retrieves_relevant_document():
    question = "Which sentence contains every letter of the English alphabet?"
    answer, sources = query_agent(question)
    assert "pangram" in answer.lower()
    assert "quick brown fox" in sources.lower()
