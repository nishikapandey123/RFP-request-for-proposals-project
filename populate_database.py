import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma
from extract_questions import extract_text_from_eligibility_section_from_link
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Constants
CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents():
    """Load and extract relevant sections from PDF documents."""
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = document_loader.load()

    relevant_docs = []
    for doc in documents:
        text = extract_text_from_eligibility_section_from_link(doc.metadata['source'])
        if text:
            relevant_docs.append(Document(page_content=text, metadata=doc.metadata))

    print(f"Loaded {len(relevant_docs)} relevant documents from {DATA_PATH}")
    return relevant_docs

def split_documents(documents):
    """Split documents into chunks for vector storage."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks")
    return chunks

def add_to_chroma(chunks):
    """Add document chunks to the Chroma database."""
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )
    
    chunks_with_ids = calculate_chunk_ids(chunks)
    print(f"Adding documents to the database: {len(chunks_with_ids)} documents.")
    
    for chunk in chunks_with_ids:
        print(f"Document ID: {chunk.metadata['id']}, Content: {chunk.page_content[:100]}...")

    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"👉 Adding {len(new_chunks)} new documents to the database")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    """Calculate unique chunk IDs for each document chunk."""
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

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    """Clear the existing Chroma database."""
    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
            print(f"Database cleared at {CHROMA_PATH}")
        except Exception as e:
            print(f"Error while clearing database: {e}")
    else:
        print("Database path does not exist, nothing to clear.")

if __name__ == "__main__":
    main()
