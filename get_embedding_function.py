from langchain_ollama import OllamaEmbeddings
import config


def get_embedding_function():
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
    return embeddings
