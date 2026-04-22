"""
bm25.py

This module implements a BM25-based keyword retrieval system for
information retrieval.

It includes functions to:
- load and preprocess data from a parquet file
- clean and normalize text fields for keyword-based retrieval
- tokenize text for BM25-based retrieval
- build or load a saved BM25 corpus and metadata store
- build a BM25 retriever for keyword-base search
- perform ranked keyword search and return structured results
  with product and review metadata

This module is used by the app and other project components to retrieve
and display relevant documents based on keyword search.
"""

import os
import re
import pickle
import pandas as pd
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever



def load_and_preprocess_data(path="data/processed/processed_scaled_sample.parquet"):
    """
    Load dataset and create combined text for BM25 retrieval.

    Parameters:
    ----------
    path : str
        Path to the processed parquet file.

    Returns:
    -------
    df : pandas.DataFrame
        Original dataframe with an additional `combined_text` column that
        merges product metadata, review information, features, description,
        and categories into a single text field for retrieval.
    """
    df = pd.read_parquet(path)

    df["combined_text"] = (
        df["product_title"].fillna("").astype(str) + ". " +
        df["review_title"].fillna("").astype(str) + ". " +
        df["review_text"].fillna("").astype(str) + ". " +
        df["features"].fillna("").astype(str) + ". " +
        df["description"].fillna("").astype(str) + ". " +
        df["categories"].fillna("").astype(str)
    )

    return df


# Simple tokenization (adapted from DSCI 575 lecture 5)
def simple_tokenize(text):
    """
    Lowercase, remove noise, and tokenize using whitespace splitting.

    Parameters:
    ----------
    text : str
        Input text to be tokenized.

    Returns:
    -------
    text : list of str
        Tokenized words from input text.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)   
    return text.split()

def get_or_build_bm25_data(data_path="data/processed/processed_scaled_sample.parquet",
                           corpus_path="data/processed/bm25_store/bm25_corpus.pkl",
                           meta_path="data/processed/bm25_store/bm25_metadata.pkl"):
    """
    Build or load BM25 corpus and metadata.

    If saved BM25 files exist, load them from disk.
    Otherwise, build the corpus and metadata from the processed dataset
    and save them for future use.

    Parameters
    ----------
    data_path : str, default="data/processed/processed_scaled_sample.parquet"
        Path to the processed parquet dataset.
    corpus_path : str, default="data/processed/bm25_store/bm25_corpus.pkl"
        File path where the tokenized corpus is saved/loaded.
    meta_path : str, default="data/processed/bm25_store/bm25_metadata.pkl"
        File path where metadata is saved/loaded.

    Returns
    -------
    corpus : list of list of str
        Tokenized BM25 corpus (each document is a list of tokens).
    metadata : list of dict
        Document metadata including product title, review text, price, etc.
    """

    if os.path.exists(corpus_path) and os.path.exists(meta_path):

        with open(corpus_path, "rb") as f:
            corpus = pickle.load(f)

        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        print("Loaded BM25 data.")
        return corpus, metadata
    
    print("Building BM25 data...")

    df = load_and_preprocess_data(data_path)

    corpus = df["combined_text"].apply(simple_tokenize).tolist()

    metadata = df[[
        "product_title",
        "review_title",
        "review_text",
        "price",
        "average_rating",
        "rating_number"
    ]].to_dict("records")

    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)

    with open(corpus_path, "wb") as f:
        pickle.dump(corpus, f)

    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    print("BM25 data built and saved.")

    return corpus, metadata


def get_bm25_retriever(corpus, metadata, k=10):
    """
    Build a BM25 retriever from tokenized corpus.

    Parameters
    ----------
    corpus : list of list of str
        Tokenized documents, where each document is a list of tokens.
    metadata : list of dict
        Metadata corresponding to each document.
    k : int, default=10
        Number of documents to retrieve.

    Returns
    -------
    BM25Retriever
        A LangChain BM25 retriever for keyword-based search.
    """
    docs = [
        Document(
            page_content=" ".join(tokens),
            metadata=metadata[i] 
        )
        for i, tokens in enumerate(corpus)
    ]

    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k

    return retriever


def bm25_search(retriever, query, top_k=5):
    """
    Perform BM25 search and return structured results.

    Parameters:
    ----------
    query : str
        User search query.
    retriever : BM25Retriever
        BM25 retriever used for keyword-based document retrieval.
    top_k : int
        Number of results to return.

    Returns:
    results : list of dict
        Ranked search results. Each result contains:
        - rank : int
        - product_title : str
        - review_text : str
        - average_rating : float or None
        - rating_number : int or None
        - price : str
        - score : float
    """

    
    tokenized_query = simple_tokenize(query)
    scores = retriever.vectorizer.get_scores(tokenized_query)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    docs = [retriever.docs[i] for i in ranked_indices]

    results = []

    for rank, idx in enumerate(ranked_indices, 1):
        doc = retriever.docs[idx]
        m = doc.metadata

        results.append({
            "rank": rank,
            "product_title": m.get("product_title"),
            "review_text": m.get("review_text"),
            "average_rating": round(float(m.get("average_rating")), 1) if pd.notna(m.get("average_rating")) else None,
            "rating_number": int(m.get("rating_number")) if pd.notna(m.get("rating_number")) else None,
            "price": str(m.get("price")) if pd.notna(m.get("price")) else "N/A",
            "score": round(float(scores[idx]), 3),
        })

    return results

