"""
build_semantic_index.py

Build a semantic vector store for semantic search.

Run:
python -m src.build_semantic_index
"""

from src.semantic import get_or_build_vectorstore
from src.config import DATA_PATH, FAISS_STORE_PATH, EMBEDDING_MODEL_NAME

def main():
    """
    Build or load the semantic vector store. 

    If the saved FAISS vector store folder is already present, the build step
    is skipped. Otherwise, the vector store is created and saved locally for
    faster loading in future runs.
    """
    
    get_or_build_vectorstore(
        data_path=DATA_PATH,
        store_path=FAISS_STORE_PATH,
        embedding_model_name=EMBEDDING_MODEL_NAME
    )


if __name__ == "__main__":
    main()