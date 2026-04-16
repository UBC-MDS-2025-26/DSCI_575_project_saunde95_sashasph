"""
semantic.py

This module implements a semantic search system using LangChain
HuggingFace embeddings and a FAISS vector store.

It includes functions to:
- load and preprocess data from a parquet file
- clean text fields for retrieval and display
- convert rows into LangChain Document objects
- build or load a saved FAISS vector store
- perform semantic search and return ranked, structured results
  with product and review metadata

This module is used by the app and other project components to retrieve
and display relevant documents based on semantic similarity.
"""

import os
import re
import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def clean_text(value):
    """
    Clean text for retrieval and display.

    This function handles missing values, removes simple HTML break tags,
    replaces non-breaking spaces, and normalizes whitespace.

    Parameters
    ----------
    value : any
        Input value to clean.

    Returns
    -------
    str
        Cleaned text string. Returns an empty string for missing values.
    """
    if pd.isna(value):
        return ""

    text = str(value)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_documents(path="data/processed/processed_data_sample.parquet"):
    """
    Load the dataset and create a combined text column for semantic retrieval.

    The combined text field merges product metadata and review content into
    one searchable string so that the embedding model has access to the
    main information used for retrieval.

    Parameters
    ----------
    path : str, default="data/processed/processed_data_sample.parquet"
        Path to the processed parquet file.

    Returns
    -------
    pandas.DataFrame
        Original dataframe with an added `combined_text` column.
    """
    df = pd.read_parquet(path)

    df["combined_text"] = (
        "Product title: " + df["product_title"].fillna("").astype(str) + ". "
        + "Review title: " + df["review_title"].fillna("").astype(str) + ". "
        + "Review text: " + df["text"].fillna("").astype(str) + ". "
        + "Features: " + df["features"].fillna("").astype(str) + ". "
        + "Description: " + df["description"].fillna("").astype(str) + ". "
        + "Categories: " + df["categories"].fillna("").astype(str)
    )

    return df


def make_langchain_documents(df):
    """
    Convert dataframe rows into LangChain Document objects.

    Each row is converted into:
    - `page_content`: the cleaned combined text used for semantic retrieval
    - `metadata`: structured fields used later by the app for display

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing `combined_text` and metadata columns.

    Returns
    -------
    list of Document
        LangChain Document objects built from the dataframe rows.
    """
    docs = []

    for _, row in df.iterrows():
        docs.append(
            Document(
                page_content=clean_text(row["combined_text"]),
                metadata={
                    "product_title": clean_text(row["product_title"]),
                    "review_title": clean_text(row["review_title"]),
                    "review_text": clean_text(row["text"]),
                    "features": clean_text(row["features"]),
                    "description": clean_text(row["description"]),
                    "categories": clean_text(row["categories"]),
                    "rating": int(row["rating"]) if pd.notna(row["rating"]) else None,
                    "average_rating": float(row["average_rating"]) if pd.notna(row["average_rating"]) else None,
                    "rating_number": int(row["rating_number"]) if pd.notna(row["rating_number"]) else None,
                    "price": str(row["price"]) if pd.notna(row["price"]) else "N/A",
                }
            )
        )

    return docs


def get_or_build_vectorstore(
    data_path="data/processed/processed_data_sample.parquet",
    store_path="data/processed/faiss_store",
    model_name="sentence-transformers/all-MiniLM-L6-v2"
):
    """
    Load a saved FAISS vector store if it exists, otherwise build and save it.

    This helps avoid recomputing embeddings every time the app or notebook
    runs. If the vector store folder is already present, it is loaded from
    disk. Otherwise, the function loads the data, converts it into LangChain
    documents, builds a FAISS vector store, saves it locally, and returns it.

    Parameters
    ----------
    data_path : str, default="data/processed/processed_data_sample.parquet"
        Path to the processed parquet file.
    store_path : str, default="data/processed/faiss_store"
        Folder path where the FAISS vector store should be saved or loaded from.
    model_name : str, default="sentence-transformers/all-MiniLM-L6-v2"
        Name of the embedding model to use.

    Returns
    -------
    FAISS
        Loaded or newly built LangChain FAISS vector store.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    if os.path.exists(store_path):
        print("Loaded semantic vector store.")
        return FAISS.load_local(
            store_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    print("No saved semantic vector store found. Building vector store (this may take a while)...")

    df = load_documents(data_path)
    docs = make_langchain_documents(df)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(store_path)

    print("Semantic setup complete!")
    return vectorstore


def semantic_search(query, vectorstore, top_k=5):
    """
    Perform semantic search and return structured, app-friendly results.

    The vector store retrieves the top-k most similar documents for the query.
    Each returned document is paired with its FAISS distance score. The
    distance is converted to a normalized similarity-style score using
    1 / (1 + distance), so that higher scores indicate more similar results.

    Parameters
    ----------
    query : str
        User search query.
    vectorstore : FAISS
        LangChain FAISS vector store used for similarity search.
    top_k : int, default=5
        Number of results to return.

    Returns
    -------
    list of dict
        Ranked search results. Each result contains:
        - rank : int
        - product_title : str
        - review_text : str
        - rating : int or None
        - average_rating : float or None
        - rating_number : int or None
        - price : str
        - score : float
    """
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=top_k)

    results = []
    for rank, (doc, distance) in enumerate(docs_with_scores, start=1):
        score = 1 / (1 + float(distance))

        results.append({
            "rank": rank,
            "product_title": doc.metadata.get("product_title", ""),
            "review_text": doc.metadata.get("review_text", ""),
            "rating": doc.metadata.get("rating"),
            "average_rating": doc.metadata.get("average_rating"),
            "rating_number": doc.metadata.get("rating_number"),
            "price": doc.metadata.get("price", "N/A"),
            "score": round(score, 3),
        })

    return results