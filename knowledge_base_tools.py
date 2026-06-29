from langchain.tools import tool
from qdrant_client import QdrantWrapper
from ollama_client import OllamaWrapper

qdrant = QdrantWrapper()
ollama = OllamaWrapper()

@tool(
    name="search_knowledge_base",
    description="Search the knowledge base for relevant documents."
)
def search_knowledge_base(query: str) -> str:
    embedding = ollama.get_embedding(query)
    results = qdrant.search(embedding, limit=3)
    if not results:
        return "No relevant documents found."
    formatted = ""
    for idx, res in enumerate(results, 1):
        text = res.payload.get("text", "")
        metadata = {k: v for k, v in res.payload.items() if k != "text"}
        formatted += f"{idx}. {text}\n   Metadata: {metadata}\n"
    return formatted

@tool(
    name="add_to_knowledge_base",
    description="Add a document to the knowledge base."
)
def add_to_knowledge_base(document: str, metadata: dict) -> str:
    embedding = ollama.get_embedding(document)
    point_id = qdrant.upsert(document, embedding, metadata)
    return f"Document added with id {point_id}."
