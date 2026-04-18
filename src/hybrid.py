"""
hybrid.py

This module implements a hybrid retrieval system that combines:
- a semantic retriever (FAISS-based vector search)
- a BM25 keyword retriever

The hybrid retriever is used in a RAG pipeline to improve retrieval quality
by leveraging both semantic similarity and keyword matching over Amazon
product reviews and metadata.
"""

from langchain_classic.retrievers import EnsembleRetriever

def get_hybrid_retriever(semantic_retriever, bm25_retriever):
    """
    Create a hybrid retriever by combining semantic and BM25 retrievers using weighted ensemble ranking.


    Parameters
    ----------
    semantic_retriever :
        Retriever that performs semantic search using vector embeddings (e.g., FAISS-based retriever).
    bm25_retriever :
        Retriever that performs keyword-based search using BM25 scoring.

    Returns
    -------
    EnsembleRetriever
        A combined retriever that merges semantic and BM25 results using weighted scoring.
    """

    ensemble = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.6, 0.4] 
    )

    return ensemble

