"""
build_semantic_index.py

Build a semantic vector store for semantic search.

Run:
python -m src.build_semantic_index
"""

import os
from src.semantic import get_or_build_vectorstore


def main():
    """
    Build the semantic vector store if it does not already exist.

    If the saved FAISS vector store folder is already present, the build step
    is skipped. Otherwise, the vector store is created and saved locally for
    faster loading in future runs.
    """
    data_path = "data/processed/processed_scaled_sample.parquet"
    store_path = "data/processed/faiss_store"

    if os.path.exists(store_path):
        print("Loaded semantic vector store.")
        return

    get_or_build_vectorstore(
        data_path=data_path,
        store_path=store_path,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


if __name__ == "__main__":
    main()