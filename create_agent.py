from langchain_community.llms import Ollama
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import Qdrant
from langchain.agents import initialize_agent, AgentType
from knowledge_base_tools import search_knowledge_base, add_to_knowledge_base
from qdrant_client import QdrantWrapper
from langchain_community.embeddings import OllamaEmbeddings
from config import OLLAMA_MODEL

def create_agent():
    qdrant_wrapper = QdrantWrapper()
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
    vector_store = Qdrant(
        client=qdrant_wrapper.client,
        collection_name=qdrant_wrapper.collection_name,
        embeddings=embeddings
    )
    memory = VectorStoreRetrieverMemory(vectorstore=vector_store)
    llm = Ollama(model=OLLAMA_MODEL)
    tools = [search_knowledge_base, add_to_knowledge_base]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        memory=memory,
        verbose=True
    )
    return agent
