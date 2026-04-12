"""
build_semantic_index.py

Build embeddings and a FAISS index for semantic search.

Run from project root:
python src/build_semantic_index.py
"""

import os
from src.semantic import (
    load_documents,
    build_embeddings,
    build_faiss_index,
    save_faiss_index
)


def main():
    data_path = "data/processed/processed_data_sample.parquet"
    index_path = "data/processed/faiss_index.index"

    # Optional: avoid rebuilding if index already exists
    if os.path.exists(index_path):
        print(f"FAISS index already exists at {index_path}. Skipping build.")
        return

    print("Loading documents...")
    df, documents = load_documents(data_path)

    print(f"Loaded {len(documents)} documents.")

    print("Building embeddings (this may take a while)...")
    model, embeddings = build_embeddings(documents)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("Saving FAISS index...")
    save_faiss_index(index, index_path)

    print(f"Done! FAISS index saved to: {index_path}")


if __name__ == "__main__":
    main()