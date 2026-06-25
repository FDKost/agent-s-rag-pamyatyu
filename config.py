from dataclasses import dataclass

@dataclass
class Config:
    # Qdrant settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "rag_collection"
    qdrant_vector_size: int = 384  # depends on embedding model
    qdrant_distance: str = "Cosine"
    in_memory: bool = False  # set to True for tests

    # Ollama settings
    ollama_model: str = "llama3"

    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"
