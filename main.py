from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from get_embedding_function import get_embedding_function
from populate_database import populate_database

CHROMA_PATH = "chroma_db"


PROMPT_TEMPLATE = """
You are a helpful and detailed assistant. You always answer based only on the provided context.

Context:
{context}

---

Question:
{question}

Give a clear, complete, and helpful answer. Include examples if helpful.
"""


def query_rag(query_text: str):
    # Prepare the DB
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    # Search the DB
    results = db.similarity_search_with_score(query_text, k=8)

    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, _score in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    prompt = prompt_template.format(
        context=context_text, question=query_text)

    model = OllamaLLM(
        model="llama3.2",
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,
        num_predict=500
    )

    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]

    return response_text, sources


def chat():
    print("🧠 PDF RAG Chat is ready! Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting chat.")
            break

        answer, sources = query_rag(user_input)
        print("\n🤖 Answer:\n", answer)
        print("\n📚 Sources:", sources, "\n")


if __name__ == "__main__":
    populate_database()
    chat()
