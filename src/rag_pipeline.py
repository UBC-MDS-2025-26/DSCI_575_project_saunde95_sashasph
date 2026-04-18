"""
rag_pipeline.py

This module implements a semantic RAG pipeline using:
- a semantic retriever built from a FAISS vector store
- a context-building function
- a prompt template
- a hosted LLM via Groq

The pipeline is used to answer user queries using retrieved Amazon
product review and metadata context.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.semantic import get_or_build_vectorstore, get_semantic_retriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SYSTEM_PROMPT_FINAL = """
You are a helpful Amazon shopping assistant for Patio, Lawn and Garden products.

Answer the user's question using ONLY the provided Amazon product review and metadata context.
Use professional and friendly language and do not make up details that are not supported by the context.
If the context is not sufficient to answer confidently, say so.

When answering:
- keep the answer concise and practical
- recommend only products supported by the retrieved context
- always give at least 3 recommendations (up to 5 if useful)
- mention product titles
- include price when it is available and not missing
- avoid repeating duplicate products
- explain why each product is a good gift in a natural way, using review insights when helpful
- focus on what makes each option appealing (e.g., practical, unique, durable, beginner-friendly)
- do NOT list raw rating statistics unless they are especially important
- do NOT include reviewer names or quote review titles
- highlight differences between options when relevant
- avoid generic phrases like "this is a great gift" without adding a specific reason
- do not include a concluding summary sentence
"""


def get_llm(model_name="llama-3.1-8b-instant"):
    """
    Load the Groq chat model used for generation.

    Parameters
    ----------
    model_name : str, default="llama-3.1-8b-instant"
        Name of the Groq-hosted model to use.

    Returns
    -------
    ChatGroq
        Initialized Groq chat model.
    """
    load_dotenv()

    if os.getenv("GROQ_API_KEY") is None:
        raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

    return ChatGroq(model=model_name)


def build_context(docs):
    """
    Convert retrieved LangChain documents into a structured context block.

    Parameters
    ----------
    docs : list
        Retrieved LangChain Document objects.

    Returns
    -------
    str
        Prompt-ready context string.
    """
    context_blocks = []

    for i, doc in enumerate(docs, start=1):
        price = doc.metadata.get("price", "N/A")
        if price in [None, "", "N/A", "nan", "NaN"]:
            price_line = ""
        else:
            price_line = f"- Price: {price}\n"

        review_text = str(doc.metadata.get("review_text", "N/A"))[:400]
        features = str(doc.metadata.get("features", "N/A"))[:250]

        block = (
            f"Document {i}\n"
            f"Product title: {doc.metadata.get('product_title', 'N/A')}\n"
            f"Review title: {doc.metadata.get('review_title', 'N/A')}\n"
            f"Review text: {review_text}\n"
            f"Review evidence:\n"
            f"- Review rating: {doc.metadata.get('rating', 'N/A')}\n"
            f"- Average product rating: {doc.metadata.get('average_rating', 'N/A')}\n"
            f"- Number of ratings: {doc.metadata.get('rating_number', 'N/A')}\n"
            f"{price_line}"
            f"- Features: {features}\n"
        )

        context_blocks.append(block)

    return "\n\n---\n\n".join(context_blocks)

def build_prompt(query, context, system_prompt=SYSTEM_PROMPT_FINAL):
    """
    Build the final RAG prompt from the system prompt, retrieved context,
    and user query.

    Parameters
    ----------
    query : str
        User question.
    context : str
        Retrieved and formatted context block.
    system_prompt : str, default=SYSTEM_PROMPT_FINAL
        System instruction for the LLM.

    Returns
    -------
    str
        Final prompt string.
    """
    return f"""{system_prompt}

Retrieved context:
{context}

User question:
{query}

Answer:"""


def get_semantic_rag_components(
    data_path=None,
    store_path=None,
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    llm_model_name="llama-3.1-8b-instant",
    top_k=5,
):
    """
    Build the main semantic RAG components.

    Parameters
    ----------
    data_path : str or Path or None, default=None
        Path to the processed parquet file. If None, the default project path is used.
    store_path : str or Path or None, default=None
        Path to the saved semantic FAISS vector store. If None, the default project path is used.
    embedding_model_name : str, default="sentence-transformers/all-MiniLM-L6-v2"
        Name of the embedding model used for semantic retrieval.
    llm_model_name : str, default="llama-3.1-8b-instant"
        Name of the Groq-hosted LLM.
    top_k : int, default=5
        Number of documents to retrieve.

    Returns
    -------
    tuple
        (vectorstore, retriever, llm)
    """
    if data_path is None:
        data_path = PROJECT_ROOT / "data" / "processed" / "processed_data_sample.parquet"

    if store_path is None:
        store_path = PROJECT_ROOT / "data" / "processed" / "faiss_store"

    vectorstore = get_or_build_vectorstore(
        data_path=str(data_path),
        store_path=str(store_path),
        model_name=embedding_model_name,
    )
    retriever = get_semantic_retriever(vectorstore, top_k=top_k)
    llm = get_llm(model_name=llm_model_name)

    return vectorstore, retriever, llm


def run_semantic_rag(query, retriever, llm, system_prompt=SYSTEM_PROMPT_FINAL):
    """
    Run the full semantic RAG pipeline:
    query -> retrieve documents -> build context -> build prompt -> generate answer

    Parameters
    ----------
    query : str
        User query.
    retriever :
        Semantic retriever that returns relevant documents.
    llm :
        Language model used for generation.
    system_prompt : str, default=SYSTEM_PROMPT_FINAL
        Prompt instructions for the LLM.

    Returns
    -------
    dict
        Dictionary containing query, retrieved docs, context, prompt, and answer.
    """
    docs = retriever.invoke(query)
    context = build_context(docs)
    prompt = build_prompt(query, context, system_prompt)
    response = llm.invoke(prompt)

    answer = response.content if hasattr(response, "content") else str(response)

    return {
        "query": query,
        "docs": docs,
        "context": context,
        "prompt": prompt,
        "answer": answer,
    }