import pandas as pd
import re
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


    
# Load dataset
df = pd.read_parquet("../data/processed/processed_data_sample.parquet")

# Clean data
df["product_title"] = df["product_title"].fillna("").astype(str)
df["review_title"] = df["review_title"].fillna("").astype(str)
df["review_text"] = df["text"].fillna("").astype(str)


# Create combined text
df["combined_text"] = (
    df["product_title"] + ". " +
    df["review_title"] + ". " +
    df["text"]
)

# Simple tokenization (adapted from DSCI 575 lecture 5)
def simple_tokenize(text):
    """
    Lowercase, remove noise, and tokenize using whitespace splitting.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)   
    return text.split()

df["tokens"] = df["combined_text"].apply(simple_tokenize)

# Create LangChain Documents
docs = [
    Document(
        page_content=row["combined_text"],
        metadata={
            "product_title": row["product_title"],
            "review_title": row["review_title"],
            "text": row["text"],
            "price": row.get("price"),
            "rating": row.get("rating"),
            "average_rating": row.get("average_rating"),
            "rating_number": row.get("rating_number"),
            "features": row.get("features"),
            "description": row.get("description"),
            "categories": row.get("categories")
        }
    )
    for _, row in df.iterrows()
]

# Create BM25 retriever
k = 10

retriever = BM25Retriever.from_documents(docs)
retriever.k = k


#for test in terminal(command:python bm25.py)
if __name__ == "__main__":

    query = "good garden tools"
    k = 10

    results = retriever.invoke(query)

    for i, doc in enumerate(results[:k], 1):
        print(f"Rank {i}")
        print("PRODUCT:", doc.metadata["product_title"])
        print("REVIEW TITLE:", doc.metadata["review_title"])
        print("REVIEW:", doc.metadata["text"])
        print("-" * 100)