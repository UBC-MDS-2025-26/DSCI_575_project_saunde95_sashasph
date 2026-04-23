"""
hybrid.py

This module implements a hybrid retrieval system that combines:
- a semantic retriever (FAISS-based vector search)
- a BM25 keyword retriever

It also provides a hybrid search function for ranking and returning
top-k retrieved documents.

The hybrid retriever is used in both search-only and RAG pipelines to
improve retrieval quality by leveraging semantic similarity and keyword
matching over Amazon product reviews and metadata.
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
        weights=[0.5, 0.5] 
    )

    return ensemble

def hybrid_search(query, hybrid_retriever, top_k=3):
    """
    Run hybrid retrieval using an ensemble retriever.

    Parameters
    ----------
    query : str
        User query.
    hybrid_retriever : EnsembleRetriever
        Combined semantic + BM25 retriever.
    top_k : int
        Number of documents to return.

    Returns
    -------
    list
        Top-k retrieved documents.
    """
    
    docs = hybrid_retriever.invoke(query)

    results = []
    for rank, doc in enumerate(docs[:top_k], start=1):
        results.append({
            "rank": rank,
            "product_title": doc.metadata.get("product_title", ""),
            "review_text": doc.metadata.get("review_text", ""),
            "average_rating": doc.metadata.get("average_rating"),
            "rating_number": doc.metadata.get("rating_number"),
            "price": doc.metadata.get("price", "N/A"),
        })

    return results
