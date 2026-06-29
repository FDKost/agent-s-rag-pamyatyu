import requests
from config import OLLAMA_HOST, OLLAMA_MODEL

class OllamaWrapper:
    def __init__(self):
        self.host = OLLAMA_HOST.rstrip("/")
        self.model = OLLAMA_MODEL

    def get_embedding(self, text: str) -> list[float]:
        url = f"{self.host}/api/embeddings"
        payload = {"model": self.model, "input": text}
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()["embeddings"][0]["embedding"]

    def generate(self, prompt: str) -> str:
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()["response"]
