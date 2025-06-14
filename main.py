from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from get_embedding_function import get_embedding_function
from populate_database import populate_database
import config
import sqlite3


def query_rag(query_text: str):
    try:
        # Prepare the DB
        db = Chroma(
            persist_directory=config.CHROMA_PATH,
            embedding_function=get_embedding_function()
        )

        # Search the DB
        results = db.similarity_search_with_score(query_text, k=8)

    except sqlite3.OperationalError as e:
        print(f"❌ Error connecting to Chroma DB: {e}")
        return None, None
    except Exception as e:  # Catch other potential Chroma errors
        print(f"❌ Error during Chroma DB operation: {e}")
        return None, None

    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, _score in results])

    prompt_template = ChatPromptTemplate.from_template(config.PROMPT_TEMPLATE)

    prompt = prompt_template.format(
        context=context_text, question=query_text)

    model = OllamaLLM(
        model=config.LLM_MODEL,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,
        num_predict=500
    )

    try:
        response_text = model.invoke(prompt)
    except Exception as e:  # Catch potential Ollama errors
        print(f"❌ Error invoking LLM: {e}")
        return None, None

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
        if answer is None:
            print("Sorry, I encountered an error. Please try again.")
            continue
        print("\n🤖 Answer:\n", answer)
        print("\n📚 Sources:", sources, "\n")


if __name__ == "__main__":
    populate_database()
    chat()
