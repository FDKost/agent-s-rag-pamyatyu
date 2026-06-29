import json
from knowledge_base_tools import add_to_knowledge_base, search_knowledge_base

def test_add_and_search():
    doc = "The quick brown fox jumps over the lazy dog."
    metadata = {"category": "animals"}
    add_msg = add_to_knowledge_base(document=doc, metadata=metadata)
    assert "added" in add_msg.lower()
    results = search_knowledge_base(query="fox")
    assert "quick brown fox" in results
