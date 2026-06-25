from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from qdrant_client import get_qdrant_client
from config import Config

def get_chain(config: Config):
    client = get_qdrant_client(config)
    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
    vectorstore = Qdrant(
        client=client,
        collection_name=config.qdrant_collection,
        embeddings=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = Ollama(model=config.ollama_model)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    return chain
