"""
semantic.py

This module implements a semantic search system using sentence embeddings
and FAISS for efficient similarity search.

It includes functions to:
- Load and preprocess documents from a parquet file
- Generate embeddings using a sentence-transformer model
- Build a FAISS index for fast nearest neighbor search
- Perform semantic search and return ranked, structured results with
  product and review metadata

This module is used by the app and other project components to retrieve
and display documents based on semantic similarity.
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def load_documents(path="../data/processed/processed_data_sample.parquet"):
    """
    Load dataset and create combined text for semantic search.

    Parameters:
    ----------
    path : str
        Path to the processed parquet file.

    Returns:
    -------
    df : pandas.DataFrame
        Original dataframe containing product and review metadata.
    documents : list of str
        List of combined text documents used for embedding and retrieval.
    """
    df = pd.read_parquet(path)

    df["combined_text"] = (
        "Product title: " + df["product_title"].fillna("").astype(str) + ". " +
        "Review title: " + df["review_title"].fillna("").astype(str) + ". " +
        "Review text: " + df["text"].fillna("").astype(str) + ". " +
        "Features: " + df["features"].fillna("").astype(str) + ". " +
        "Description: " + df["description"].fillna("").astype(str) + ". " +
        "Categories: " + df["categories"].fillna("").astype(str)
    )

    documents = df["combined_text"].str.strip().tolist()
    return df, documents


def build_embeddings(documents, model_name="all-MiniLM-L6-v2"):
    """
    Generate embeddings for documents using a sentence-transformer model.

    Parameters:
    ----------
    documents : list of str
        Text documents to embed.
    model_name : str
        Name of the sentence-transformer model.

    Returns:
    -------
    model : SentenceTransformer
        Loaded embedding model.
    embeddings : numpy.ndarray
        Array of document embeddings.
    """
    if not documents:
        raise ValueError("No documents found.")

    model = SentenceTransformer(model_name)
    embeddings = model.encode(documents, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")
    return model, embeddings


def build_faiss_index(embeddings):
    """
    Build a FAISS index from document embeddings.

    Parameters:
    ----------
    embeddings : numpy.ndarray
        Array of document embeddings.

    Returns:
    -------
    index : faiss.Index
        FAISS index for similarity search.
    """
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def semantic_search(query, model, index, df, top_k=5):
    """
    Perform semantic search using FAISS and return structured results.

    Parameters:
    ----------
    query : str
        User search query.
    model : SentenceTransformer
        Embedding model used to encode the query.
    index : faiss.Index
        FAISS index containing document embeddings.
    df : pandas.DataFrame
        DataFrame containing original product and review metadata.
    top_k : int
        Number of results to return.

    Returns:
    -------
    results : list of dict
        Ranked search results. Each result contains:
        - rank : int
        - product_title : str
        - review_text : str
        - rating : int or None
        - average_rating : float or None
        - rating_number : int or None
        - price : str
        - distance : float (lower = more similar)
    """
    top_k = min(top_k, len(df))

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding, dtype="float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for rank, doc_idx in enumerate(indices[0]):
        row = df.iloc[doc_idx]

        results.append({
            "rank": rank + 1,
            "product_title": row["product_title"],
            "review_text": row["text"],
            "rating": int(row["rating"]) if pd.notna(row["rating"]) else None,
            "average_rating": float(row["average_rating"]) if pd.notna(row["average_rating"]) else None,
            "rating_number": int(row["rating_number"]) if pd.notna(row["rating_number"]) else None,
            "price": str(row["price"]) if pd.notna(row["price"]) else "N/A",
            "distance": float(distances[0][rank])
        })

    return results

def save_faiss_index(index, path="../data/processed/faiss_index.index"):
    """
    Save a FAISS index to disk.

    Parameters:
    ----------
    index : faiss.Index
        FAISS index to save.
    path : str
        Output path for the saved FAISS index.
    """
    faiss.write_index(index, path)


def load_faiss_index(path="../data/processed/faiss_index.index"):
    """
    Load a FAISS index from disk.

    Parameters:
    ----------
    path : str
        Path to a saved FAISS index.

    Returns:
    -------
    index : faiss.Index
        Loaded FAISS index.
    """
    return faiss.read_index(path)