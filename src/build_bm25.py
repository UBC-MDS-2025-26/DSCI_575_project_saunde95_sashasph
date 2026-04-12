"""
build_bm25.py

Build BM25 search system.

Run:
python src/main_bm25.py
"""
import sys
from src.bm25 import (
    load_and_preprocess_data,
    add_tokens,
    create_langchain_docs,
    build_bm25_retriever,
    bm25_search,
    save_documents,
    load_documents,
    save_bm25_retriever,
    load_bm25_retriever
)


def main():

    data_path = "data/processed/processed_data_sample.parquet"

    print("Loading documents...")
    docs = load_documents()
    
    if docs is None:
        print("No saved documents found. Creating documents (this may take a while)...")
        df = load_and_preprocess_data(data_path)
        df = add_tokens(df)
        docs = create_langchain_docs(df)
        save_documents(docs)
        print("Documents saved!")

    else:
        print("Loaded documents.")

    retriever = load_bm25_retriever()

    if retriever is None:
        print("No saved BM25 retriever found. Building retriever (this may take a while)...")

        retriever = build_bm25_retriever(docs, k=10)

        save_bm25_retriever(retriever)

    else:
        print("Loaded BM25 retriever.")

    print("BM25 setup complete!")

if __name__ == "__main__":
    main()
