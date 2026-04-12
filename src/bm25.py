"""
bm25.py

This module implements a bm25-based search system for information retrieval.

It includes functions to:
- Load and preprocess data from a parquet file
- Tokenize text for keyword-based retrieval
- Convert data into LangChain Document objects
- Build a BM25 retriever
- Perform BM25 search and return ranked, structured results with
  product and review metadata

This module is used by the app and other project components to retrieve
and display relevant documents based on keyword search.
"""

import pandas as pd
import re
import pickle
import os
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever



def load_and_preprocess_data(path="data/processed/processed_data_sample.parquet"):
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
        df["text"].fillna("").astype(str) +
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


def add_tokens(df):
    """
    Apply tokenization to a dataframe column and store results in a new column.

    Parameters:
    ----------
    df : pandas.DataFrame
        Input dataframe containing text data.

    Returns:
    -------
    df: pandas.DataFrame
        DataFrame with an added `tokens` column.
    """
    df["tokens"] = df["combined_text"].apply(simple_tokenize)

    return df


def create_langchain_docs(df):
    """
    Convert a pandas DataFrame into LangChain Document objects for BM25 retrieval.

    Parameters:
    ----------
    df : pandas.DataFrame
        Input data for document creation.

    Returns:
    -------
    documents : list of Document
        LangChain Document objects used for retrieval.
    """
    documents = [
        Document(
            page_content=" ".join(row["tokens"]),
            metadata={
                "product_title": row["product_title"],
                "review_title": row["review_title"],
                "review_text": row["text"],
                "price": row.get("price"),
                "rating": row.get("rating"),
                "average_rating": row.get("average_rating"),
                "rating_number": row.get("rating_number"),
                "features": row.get("features"),
                "description": row.get("description"),
                "categories": row.get("categories"),
            },
        )
        for _, row in df.iterrows()
    ]

    return documents


def save_documents(documents, path="data/processed/bm25_docs.pkl"):
    """
    Save LangChain documents to disk.
    """
    with open(path, "wb") as f:
        pickle.dump(documents, f)

def load_documents(path="data/processed/bm25_docs.pkl"):
    """
    Load LangChain documents from disk.
    """
    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        return pickle.load(f)

def build_bm25_retriever(documents, k=10):
    """
    Build a BM25 retriever from a list of LangChain documents.

    Parameters:
    ----------
    documents : list of Document
        Input documents used for BM25 indexing and retrieval.
    k : int
        Number of top results to return during retrieval.

    Returns:
    -------
    retriever : BM25Retriever
        Configured BM25 retriever instance.
    """
    if not documents:
        raise ValueError("No documents found.")

    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k

    return retriever


def save_bm25_retriever(retriever, path="data/processed/bm25_retriever.pkl"):
    """
    Save BM25 retriever to disk.
    """
    with open(path, "wb") as f:
        pickle.dump(retriever, f)


def load_bm25_retriever(path="data/processed/bm25_retriever.pkl"):
    """
    Load BM25 retriever from disk.
    """
    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        return pickle.load(f)


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
        - rating : int or None
        - average_rating : float or None
        - rating_number : int or None
        - price : str
    """
    processed_query = " ".join(simple_tokenize(query)) 
    docs = retriever.invoke(processed_query)[:top_k]

    results = []

    for rank, doc in enumerate(docs, 1):
        m = doc.metadata

        results.append({
            "rank": rank,
            "product_title": m.get("product_title"),
            "review_text": m.get("review_text"),
            "rating": m.get("rating"),
            "average_rating": m.get("average_rating"),
            "rating_number": m.get("rating_number"),
            "price": m.get("price")
        })

    return results

