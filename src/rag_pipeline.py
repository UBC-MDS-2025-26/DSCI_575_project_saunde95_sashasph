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
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.semantic import get_or_build_vectorstore, get_semantic_retriever
from src.config import DATA_PATH, FAISS_STORE_PATH, LLM_MODEL_NAME, EMBEDDING_MODEL_NAME


SYSTEM_PROMPT_FINAL = """
You are a helpful Amazon shopping assistant for Patio, Lawn and Garden products.

Answer the user's question using ONLY the provided Amazon product review and metadata context.
Do NOT make up products or details that are not supported by the context.
If the context is not sufficient, clearly say so.

FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:

- Provide up to 5 product recommendations when relevant.
- prioritize products that best match the user’s request
- use review insights to support explanations when helpful (e.g., durability, ease of use, common issues)
- avoid generic or repetitive statements about ratings (e.g., "highly rated", "4.5 stars") unless they add meaningful context
- do not include products that clearly do not match key constraints in the query (e.g., wrong color, size, or use)
- only include products that are supported by the context and are meaningfully relevant; do not include filler items to reach a specific number
- Use a clean numbered list (1., 2., 3., etc.) with no extra text between items
- Each item must follow this structure:

<Number>. **Product Title** (Price if available)
1–2 sentences of why this product fits the user’s request.

Optionally include a short one-line header before the list that directly reflects the user’s request.
Do NOT include long introductions or explanations before the list.
Do NOT include a concluding summary sentence.

STYLE GUIDELINES:
- Keep the tone professional and friendly
- Be concise and practical
- Avoid generic phrases (e.g., “great gift”) without specific reasoning
- Highlight what makes each option distinct when relevant
- Avoid duplicate or irrelevant products
"""


def get_llm(model_name=LLM_MODEL_NAME):
    """
    Load the Groq chat model used for generation.

    Parameters
    ----------
    model_name : str, default=LLM_MODEL_NAME
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
    data_path=DATA_PATH,
    store_path=FAISS_STORE_PATH,
    embedding_model_name=EMBEDDING_MODEL_NAME,
    llm_model_name=LLM_MODEL_NAME,
    top_k=5,
):
    """
    Build the main semantic RAG components.

    Parameters
    ----------
    data_path : str or pathlib.Path, default=DATA_PATH
        Path to the processed parquet file. 
    store_path : str or pathlib.Path, default=FAISS_STORE_PATH
        Path to the saved semantic FAISS vector store. 
    embedding_model_name : str, default=EMBEDDING_MODEL_NAME
        Name of the embedding model used for semantic retrieval.
    llm_model_name : str, default=LLM_MODEL_NAME
        Name of the Groq-hosted LLM.
    top_k : int, default=5
        Number of documents to retrieve.

    Returns
    -------
    tuple
        (vectorstore, retriever, llm)
    """

    vectorstore = get_or_build_vectorstore(
        data_path=data_path,
        store_path=store_path,
        embedding_model_name=embedding_model_name,
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