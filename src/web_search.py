"""
web_search.py
"""

from tavily import TavilyClient
from langchain.tools import tool
import os
from dotenv import load_dotenv





def get_tavily_client():
    """
    Initialize and return a Tavily API client for web search.

    Returns
    -------
    TavilyClient
        An initialized Tavily client used to perform web searches.

    Raises
    ------
    ValueError
        If the TAVILY_API_KEY is not set in the environment.
    """
    load_dotenv()

    api_key = os.getenv("TAVILY_API_KEY")

    if api_key is None:
        raise ValueError("TAVILY_API_KEY not found. Please set it in your .env file.")

    return TavilyClient(api_key=api_key)


@tool
def web_search(query, max_results=3):
    """
    Use the web_search tool ONLY when:
    - the answer is not in the provided context
    - or the user asks for current or external information

    Search the web for up-to-date product information such as:
    - current price
    - specifications
    - recent updates

    Parameters
    ----------
    query : str
        The search query.
    max_results : int
        Number of results to return.

    Returns
    -------
    str
        Concatenated snippets from search results.
    """
    tavily_client = get_tavily_client()

    results = tavily_client.search(query, max_results=max_results)
    snippets = [r["content"] for r in results.get("results", [])]

    return "\n".join(snippets)