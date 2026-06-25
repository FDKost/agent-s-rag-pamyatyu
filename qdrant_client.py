from qdrant_client import QdrantClient
from config import Config

def get_qdrant_client(config: Config) -> QdrantClient:
    return QdrantClient(
        host=config.qdrant_host,
        port=config.qdrant_port,
        prefer_grpc=False,
        timeout=5,
        timeout_retry=5,
        timeout_retry_interval=1,
        timeout_retry_max_attempts=3,
        timeout_retry_max=5,
        in_memory=config.in_memory
    )

def create_collection(client: QdrantClient, config: Config):
    client.recreate_collection(
        collection_name=config.qdrant_collection,
        vectors_config={"size": config.qdrant_vector_size, "distance": config.qdrant_distance}
    )
