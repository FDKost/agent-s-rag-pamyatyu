from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from qdrant_client.http import models
from config import QDRANT_URL, QDRANT_API_KEY, OLLAMA_MODEL, OLLAMA_BASE_URL
from utils import read_text_files
import os

COLLECTION_NAME = "rag_collection"

def get_qdrant_client() -> QdrantClient:
    if QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(url=QDRANT_URL)

def create_collection(client: QdrantClient):
    if COLLECTION_NAME not in client.get_collections().collections:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=768,
                distance=models.Distance.COSINE
            ),
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=200
            )
        )

def load_and_index_documents():
    client = get_qdrant_client()
    create_collection(client)

    embeddings = OllamaEmbeddings(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    texts = read_text_files("sample_data")
    docs = []
    for text in texts:
        chunks = splitter.split_text(text)
        docs.extend(chunks)

    vectorstore = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings
    )

    vectorstore.add_texts(
        texts=docs,
        ids=[str(i) for i in range(len(docs))],
        metadatas=[{"source": "sample_data"} for _ in docs]
    )

def get_retrieval_chain():
    client = get_qdrant_client()
    embeddings = OllamaEmbeddings(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    vectorstore = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = Ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return chain

def query_agent(question: str):
    chain = get_retrieval_chain()
    result = chain({"query": question})
    answer = result.get("result", "")
    sources = result.get("source_documents", [])
    source_texts = "\n---\n".join([doc.page_content for doc in sources])
    return answer, source_texts
