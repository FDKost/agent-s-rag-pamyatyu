# RAG Agent with Qdrant and Ollama

This project demonstrates a Retrieval-Augmented Generation (RAG) agent that uses **Qdrant** for vector storage and **Ollama** for embeddings and LLM inference. The agent can index documents, store their embeddings, and answer queries using retrieved context.

## Prerequisites

- Python 3.10+
- Docker (for running Qdrant locally)
- Ollama installed locally (or accessible via a remote server)

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/rag-agent.git
   cd rag-agent
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   Copy `.env.example` to `.env` and adjust values if necessary.

   ```bash
   cp .env.example .env
   ```

4. **Run Qdrant**

   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

5. **Run Ollama**

   ```bash
   ollama serve
   ```

   Ensure the model specified in `OLLAMA_MODEL` is available (`llama3` by default).

## Usage

### Index Documents

The sample data is located in `sample_data/`. To index these documents:

```bash
python main.py index
```

### Query the Agent

```bash
python main.py query "What is the capital of France?"
```

The agent will return an answer and the source documents used.

## Testing

Run unit tests with:

```bash
pytest
```

## Project Structure

- `main.py` – CLI entry point.
- `agent.py` – Core logic for indexing and querying.
- `utils.py` – Helper functions.
- `config.py` – Loads environment variables.
- `sample_data/` – Sample documents for indexing.
- `tests/` – Unit tests.

## License

MIT License
