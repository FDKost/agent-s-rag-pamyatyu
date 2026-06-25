import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import get_qdrant_client, create_collection
from config import Config

def ingest_documents(directory: str, config: Config):
    client = get_qdrant_client(config)
    create_collection(client, config)

    # Load all .txt files from the directory
    raw_texts = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                raw_texts.append(f.read())

    # Split texts into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text("\n".join(raw_texts))

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)

    # Upsert into Qdrant
    Qdrant.from_texts(
        texts=chunks,
        embedding=embeddings,
        client=client,
        collection_name=config.qdrant_collection
    )
