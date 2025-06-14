import os
import argparse
import shutil
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function
import config


def populate_database():
    if not os.path.exists(config.CHROMA_PATH):
        print("✨ Creating new Chroma database...")

        documents = PyPDFDirectoryLoader(config.DATA_PATH).load()

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        ).split_documents(documents)

        # Add IDs to chunks
        last_page_id = None
        current_chunk_index = 0
        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0
            chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

        db = Chroma(
            persist_directory=config.CHROMA_PATH,
            embedding_function=get_embedding_function()
        )
        ids = [chunk.metadata["id"] for chunk in chunks]
        print(
            f"🔄 Adding {len(chunks)} chunks to Chroma (this might take a moment)...")
        db.add_documents(chunks, ids=ids)

        print(f"✅ Added {len(chunks)} chunks to Chroma.")
    else:
        print("✅ Chroma database already exists. Checking for new documents...")
        db = Chroma(
            persist_directory=config.CHROMA_PATH,
            embedding_function=get_embedding_function()
        )
        existing_ids = set(db.get(include=[])["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        documents = PyPDFDirectoryLoader(config.DATA_PATH).load()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        ).split_documents(documents)

        chunks_with_ids = calculate_chunk_ids(chunks)

        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            print(f"👉 Adding {len(new_chunks)} new documents")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
            # db.persist()
            print(f"✅ Added {len(new_chunks)} new chunks to Chroma.")
        else:
            print("✅ No new documents to add.")


def calculate_chunk_ids(chunks):
    # This will create IDs like "data\\tutorial.pdf:6:2"
    # Page Source : Page Number : Chunk Index
    last_page_id = None
    current_chunk_index = 0
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id
    return chunks


def clear_database():
    if os.path.exists(config.CHROMA_PATH):
        print("✨ Clearing Database")
        shutil.rmtree(config.CHROMA_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        clear_database()

    populate_database()
