import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shiny import App, ui, render, reactive
from sentence_transformers import SentenceTransformer
from src.semantic import (
    load_documents as load_semantic_docs,
    load_faiss_index,
    semantic_search
)
from src.bm25 import (
    load_documents as load_bm25_docs,
    load_bm25_retriever,
    bm25_search
)

# ---- Load semantic search artifacts once at startup ----
DATA_PATH = "data/processed/processed_data_sample.parquet"
INDEX_PATH = "data/processed/faiss_index.index"
DOCS_PATH = "data/processed/bm25_docs.pkl"
BM25_PATH = "data/processed/bm25_index.pkl"

df, _ = load_semantic_docs(DATA_PATH)
index = load_faiss_index(INDEX_PATH)
model = SentenceTransformer("all-MiniLM-L6-v2")
docs = load_bm25_docs(DOCS_PATH)
bm25 = load_bm25_retriever(BM25_PATH)


def truncate_text(text, max_chars=200):
    """Truncate long review text for cleaner app display."""
    if text is None:
        return ""
    text = str(text)
    return text if len(text) <= max_chars else text[:max_chars] + "..."

def display_text(val):
    return "N/A" if val is None else str(val)

TITLE_STYLE = (
    "font-family:'Poppins', sans-serif; "
    "color:#122b15; "
    "font-size:3.6em; "
    "font-weight:900; "
    "margin:10px 0 0 0; "
    "letter-spacing:1px;"
)
SUBTITLE_STYLE = (
    "font-family:'Poppins', sans-serif; "
    "color:rgba(68,74,34,0.75); "
    "font-size:1.5em; "
    "font-weight:400; "
    "margin:4px 0 18px 0; "
    "letter-spacing:0.2px;"
)

app_ui = ui.page_fluid(
    ui.tags.div("Smart Amazon Product Query Assistant", style = TITLE_STYLE),
    ui.tags.div(
        "Search Amazon product reviews using BM25 or semantic search.",
        style = SUBTITLE_STYLE   
    ),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_text("query", "Enter your query:", placeholder="e.g. large decorative mailbox cover"),
            ui.input_radio_buttons(
                "method",
                "Search method:",
                choices=["BM25", "Semantic"],
                selected="BM25"
            ),
            ui.input_action_button("search_btn", "Search"),
        ),
        ui.output_ui("results_ui"),
    ),
)


def server(input, output, session):

    @reactive.calc
    @reactive.event(input.search_btn)
    def search_results():
        query = input.query().strip()
        method = input.method()

        if not query:
            return {"method": method, "results": [], "message": "Please enter a query."}

        if method == "Semantic":
            results = semantic_search(query, model, index, df, top_k=3)
            return {"method": method, "results": results, "message": None}

        elif method == "BM25":
            results = bm25_search(bm25, query, top_k=3)
            return {"method": method, "results": results, "message": None}

        return {"method": method, "results": results, "message": None}

    @output
    @render.ui
    def results_ui():
        search_output = search_results()
        method = search_output["method"]
        results = search_output["results"]
        message = search_output["message"]

        if message:
            return ui.div(ui.p(message))

        if not results:
            return ui.div(ui.p("No results found."))

        cards = []

        for result in results:
            rating = result.get("rating")
            average_rating = result.get("average_rating")
            rating_number = result.get("rating_number")
            price = result.get("price")
            score = result.get("score")
            review_text = truncate_text(result.get("review_text", ""), max_chars=200)

            card = ui.card(
                ui.div(
                    ui.h4(f"Rank {result['rank']}"),
                    ui.span(
                        f"{method} score: {score:.3f}" if score is not None else "N/A",
                        style="font-weight:bold;"
                        ),
                        style="display:flex; justify-content:space-between; align-items:center;"
                        ),
                        ui.h4(result.get("product_title", "Untitled product")),
                        ui.p(
                            f"Average rating: {display_text(average_rating)} | "
                            f"Number of ratings: {display_text(rating_number)} | "
                            f"Price: {display_text(price)}"
                            ),
                        ui.p(
                            f"Rating: {display_text(rating)}"
                            ),
                        ui.p(review_text),

                        full_screen=False,
                            )
            
            
            cards.append(card)

        return ui.TagList(*cards)


app = App(app_ui, server)