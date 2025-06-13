# 📄 PDF RAG Chatbot with LangChain + Ollama

A simple and local Retrieval-Augmented Generation (RAG) chatbot built with **LangChain**, **Ollama**, and **ChromaDB**. It lets you chat with any set of PDFs and get accurate answers using LLMs running on your machine.

---

## 🚀 Features

- 🔍 Ask questions about one or more PDF documents
- 📚 Uses local embeddings (Ollama + ChromaDB)
- 🤖 Powered by Ollama’s `llama3` model
- 🧠 Intelligent chunking of documents with metadata
- 💾 Fully local — no external API calls

---

## 🛠️ Tech Stack

- [LangChain](https://github.com/langchain-ai/langchain)
- [Ollama](https://ollama.com/)
- [ChromaDB](https://www.trychroma.com/)
- [uv (fast Python package manager)](https://github.com/astral-sh/uv)
- [Python 3.10+](https://www.python.org/)

---

## ⚙️ Setup Instructions

- Place your PDF files inside the data/ folder.

- Run `uv run python main.py` to start the chatbot

---

## 📝 Tips

- To reset the database, delete the chroma_db/ folder
- Make sure your populate_database() script is up-to-date with the latest LangChain changes
- If you're using llama3, make sure the model is running via Ollama
