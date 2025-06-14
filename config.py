CHROMA_PATH = "chroma_db"
DATA_PATH = "data"
LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "mxbai-embed-large"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80
PROMPT_TEMPLATE = """
You are a helpful and detailed assistant. You always answer based only on the provided context.

Context:
{context}

---

Question:
{question}

Give a clear, complete, and helpful answer. Include examples if helpful.
"""
