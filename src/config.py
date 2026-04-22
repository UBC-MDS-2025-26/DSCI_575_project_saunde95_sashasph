from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_scaled_sample.parquet"
FAISS_STORE_PATH = PROJECT_ROOT / "data" / "processed" / "faiss_store"
BM25_CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "bm25_store" / "bm25_corpus.pkl"
BM25_META_PATH = PROJECT_ROOT / "data" / "processed" / "bm25_store" / "bm25_metadata.pkl"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama-3.3-70b-versatile"