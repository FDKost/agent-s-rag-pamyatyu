import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from config import QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY, QDRANT_COLLECTION

class QdrantWrapper:
    def __init__(self):
        self.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY
        )
        self.collection_name = QDRANT_COLLECTION
        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections()
        if self.collection_name not in [c.name for c in collections.collections]:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE
                )
            )

    def upsert(self, text: str, embedding: list[float], metadata: dict):
        point_id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=metadata | {"text": text}
                )
            ]
        )
        return point_id

    def search(self, embedding: list[float], limit: int = 5):
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        return results
