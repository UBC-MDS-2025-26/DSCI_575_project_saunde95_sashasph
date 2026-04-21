"""
rag_pipeline_hybrid.py

This module implements a hybrid RAG pipeline using:
- a combined semantic and BM25 retriever (FAISS + keyword-based search)
- a context-building function
- a prompt template
- a hosted LLM via Groq

The pipeline answers user queries using combined semantic and BM25 retrieval
over Amazon product reviews and metadata.
"""

from src.rag_pipeline import build_context, build_prompt, SYSTEM_PROMPT_FINAL


def run_hybrid_rag(query, hybrid_retriever, llm):
    """
    Run the hybrid RAG pipeline:
    query -> retrieve documents -> build context -> build prompt -> generate answer

    Parameters
    ----------
    query : str
        User query.
    hybrid_retriever :
        Combined retriever (semantic + BM25) that returns relevant documents.
    llm :
        Langauge model used for generation.

    Returns
    -------
    dict
        Dictionary containing query, retrieved docs, context, prompt, and answer.
    """

    docs = hybrid_retriever.invoke(query)
    context = build_context(docs)

    if len(docs) == 0 or "N/A" in context or context.strip() == "":
        web_info = web_search(query)
        context += "\n\nWeb search results:\n" + web_info

    prompt = build_prompt(query, context, SYSTEM_PROMPT_FINAL)
    response = llm.invoke(prompt)

    answer = response.content if hasattr(response, "content") else str(response)

    return {
        "query": query,
        "docs": docs,
        "context": context,
        "prompt": prompt,
        "answer": answer,
    }