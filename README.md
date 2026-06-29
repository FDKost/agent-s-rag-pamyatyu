# RAG Agent with Qdrant and Ollama

This project implements a Retrieval-Augmented Generation (RAG) agent that uses **Qdrant** for vector storage and **Ollama** for embeddings and LLM inference. The agent can add documents to a knowledge base and answer queries by searching the stored embeddings.

## Features

- **Qdrant integration** – Stores and retrieves vector embeddings.
- **Ollama integration** – Generates embeddings and text responses.
- **RAG memory** – Uses LangChain’s `VectorStoreRetrieverMemory`.
- **Tools** – `search_knowledge_base` and `add_to_knowledge_base` are exposed as LangChain tools.
- **Agent factory** – `create_agent()` returns a fully configured LangChain agent.

## Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/rag-agent.git
cd rag-agent

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root (or set the variables in your shell):

```dotenv
# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=  # leave empty if no key
QDRANT_COLLECTION=knowledge_base

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3
```

Ensure that both Qdrant and Ollama are running locally or adjust the URLs accordingly.

## Usage

### Adding a Document

```bash
python run_agent.py --add --doc "Your document text here" --meta '{"author":"John"}'
```

### Querying the Agent

```bash
python run_agent.py --query "What is the quick brown fox?"
```

The agent will automatically use the `search_knowledge_base` tool to retrieve relevant documents and generate an answer.

## Testing

Run the tests with:

```bash
pytest
```

Make sure Qdrant and Ollama are accessible; otherwise, the tests that depend on them will be skipped.

## License

MIT License
