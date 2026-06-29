import pytest
from ollama_client import OllamaWrapper

@pytest.mark.skipif(
    not pytest.config.getoption("--ollama"),
    reason="Ollama server not available"
)
def test_embedding_and_generate():
    wrapper = OllamaWrapper()
    embedding = wrapper.get_embedding("Hello world")
    assert isinstance(embedding, list)
    response = wrapper.generate("Say hello")
    assert isinstance(response, str)
    assert "hello" in response.lower()
