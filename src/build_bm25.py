"""
build_bm25.py

Build BM25 corpus and metadata store for BM25 search

Run:
python -m src.build_bm25
"""
import os
from src.bm25 import get_or_build_bm25_data
from src.config import DATA_PATH, BM25_CORPUS_PATH, BM25_META_PATH

def main():
    """
    Build BM25 corpus and metadata if not exists, otherwise load existing data.

    If the saved BM25 store folder is already present, the build step
    is skipped. Otherwise, the corpus and metadata store is created and saved locally for
    faster loading in future runs.
    """

    data_path = DATA_PATH
    corpus_path = BM25_CORPUS_PATH
    meta_path = BM25_META_PATH
    
    if os.path.exists(corpus_path) and os.path.exists(meta_path):
        print("Loaded BM25 corpus and metadata.")
        return
    
    get_or_build_bm25_data(
        data_path=data_path,
        corpus_path=corpus_path,
        meta_path=meta_path
    )

    print("BM25 corpus and metadata built and saved.")


if __name__ == "__main__":
    main()
