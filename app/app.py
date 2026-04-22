import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from shiny import App, ui, render, reactive
from src.semantic import (
    get_or_build_vectorstore,
    get_semantic_retriever,
    semantic_search
)
from src.bm25 import (
    get_or_build_bm25_data,
    get_bm25_retriever,
    bm25_search
)
from src.hybrid import (
    get_hybrid_retriever,
    hybrid_search

)
from src.rag_pipeline import get_llm, run_semantic_rag
from src.rag_pipeline_hybrid import run_hybrid_rag


DATA_PATH = "data/processed/processed_scaled_sample.parquet"
FAISS_STORE_PATH = "data/processed/faiss_store"
BM25_STORE_PATH = "data/processed/bm25_store"

vectorstore = get_or_build_vectorstore(data_path=DATA_PATH, store_path=FAISS_STORE_PATH)
semantic = get_semantic_retriever(vectorstore, top_k=5)
corpus, metadata = get_or_build_bm25_data(data_path=DATA_PATH)
bm25 = get_bm25_retriever(corpus, metadata)
hybrid_retriever = get_hybrid_retriever(semantic, bm25)
llm = get_llm()


def truncate_text(text, max_chars=200):
    """Truncate long review text for cleaner app display."""
    if text is None:
        return ""
    text = str(text)
    return text if len(text) <= max_chars else text[:max_chars] + "..."

def display_text(val):
    return "N/A" if val is None else str(val)

def rating_to_stars(rating):
    if rating is None:
        return "N/A"

    rating = float(rating)
    full = int(rating)
    empty = 5 - full

    stars = "★" * full + "☆" * empty

    return ui.span(
        stars,
        " ",
        f"{rating:.1f}",
        style="font-size: 16px;"
    )

TITLE_STYLE = (
    "font-family:'Poppins', sans-serif; "
    "color:#ffffff; "
    "font-size:3.6em; "
    "font-weight:900; "
    "margin:10px 0 0 0; "
    "letter-spacing:1px;"
)
SUBTITLE_STYLE = (
    "font-family:'Poppins', sans-serif; "
    "color:#ffffff; "
    "font-size:1.6em; "
    "font-weight:400; "
    "margin:-5px 0 18px 0; "
    "letter-spacing:0.2px;"
)

app_ui = ui.page_fluid(
   ui.tags.head(
    ui.tags.style("""
        /* THE SEARCH BOX */
        #query {
            font-size: 1.3rem !important; 
            padding: 15px !important;
            border: 3px solid #007bff !important;
            border-radius: 12px !important;
            background-color: white !important;
            color: black !important;
            min-height: 120px !important;
            font-weight: bold !important;
        }

        /* MODE OPTION INDENT */
        #mode .shiny-options-group {
            margin-left: 35px !important; 
            font-weight: bold !important;
        }

        /* FONT SIZES & BOLDING */
        /* Main Labels & Mode Options at 1.3rem */
        .sidebar .control-label, 
        #mode .radio,
        #search_btn {
            font-size: 1.3rem !important;
            font-weight: bold !important;
        }

        /* Method Options (Semantic/BM25) */
        .indented-group .radio {
            font-size: 1.3rem !important;
            font-weight: bold !important;
        }

        /* MAIN INDENTATION (Mode Options) */
        #mode .shiny-options-group {
            margin-left: 35px !important; 
        }

        /* THE NESTED METHOD SECTION */
        .indented-group {
            margin-left: 80px !important; 
            padding-left: 15px;
            border-left: 4px solid rgba(0, 123, 255, 0.7);
            margin-top: -5px !important; 
            margin-bottom: 25px;
        }

        .indented-group .shiny-options-group {
            margin-left: 15px !important; 
            margin-top: 0px !important;
        }

        .indented-group .radio {
            font-size: 1.2rem !important;
            margin-bottom: 8px !important;
        }

        /* SEARCH BUTTON */
        #search_btn {
            height: 50px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            margin-top: 15px !important;
            border-radius: 12px !important;
            width: 100%;
        }

        /* SPACING ADJUSTMENTS */
        label[for="query"] { margin-bottom: 10px !important; margin-top: 10px !important; }
        label[for="mode"] { margin-bottom: 15px !important; margin-top: 25px !important; }

        /* BACKGROUND & SIDEBAR BASE */
        body {
            background-image: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)),
                              url('/background.jpg');
            background-size: cover;
            background-position: left;
            background-attachment: fixed;
        }
        .sidebar { color: white !important; }
    """)
),
    
    ui.tags.div("Amazon Patio, Lawn and Garden Product Search", style=TITLE_STYLE),
    ui.tags.div(
        "Explore product reviews using keyword (BM25), semantic, or hybrid search with AI-powered insights in RAG mode.",
        style=SUBTITLE_STYLE
    ),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_text_area(
                "query", 
                "Enter your query:", 
                placeholder="e.g. large decorative mailbox cover",
                rows=5,        
                width="100%"
            ),
            ui.input_radio_buttons(
                "mode",
                "Mode",
                choices=["RAG Mode", "Search Only"],
                selected="RAG Mode"
            ),
           
            ui.output_ui("method_ui_container"),
            
            ui.input_action_button("search_btn", "Search", class_="btn-primary w-100"),

            width=400
        ),
        ui.output_ui("results_ui"),
    ),
)


def server(input, output, session):

    @reactive.calc
    @reactive.event(input.search_btn)
    def search_results():
        query = input.query().strip()
        mode = input.mode()
        
        if not query:
            return {
            "mode": mode,
            "results": [],
            "answer": None,
            "docs": None,
            "message": "Please enter a query.",
            "method": None
        }

        if mode == "Search Only":
            method = input.method()

            if method == "Semantic":
                results = semantic_search(query, vectorstore, top_k=10)

            elif method == "BM25":
                results = bm25_search(bm25, query, top_k=10)
            
            elif method == "Hybrid":
                results = hybrid_search(query, hybrid_retriever, top_k=10)

            return {
                "mode": mode,
                "method": method,
                "results": results,
                "answer": None,
                "docs": None,
                "message": None
        }
        
        elif mode == "RAG Mode":
            method = input.method()

            if method == "Semantic":
                result = run_semantic_rag(
                    query=query,
                    retriever=semantic, 
                    llm=llm
                )
            
            elif method == "Hybrid":
                result = run_hybrid_rag(
                query=query,
                hybrid_retriever=hybrid_retriever,
                llm=llm
            )

            return {
                "mode": mode,
                "method": None,
                "results": None,
                "answer": result["answer"],
                "docs": result["docs"],
                "message": None
            }
        
        
    @render.ui
    def method_ui_container():
            mode = input.mode()
        
            if mode == "Search Only":
                choices = ["BM25", "Semantic", "Hybrid"]
            
                return ui.div(
                    ui.input_radio_buttons(
                        "method",
                        "Search Method",
                        choices=choices,
                        selected="Hybrid"
                    ),
                    class_="indented-group"
                )
            
            elif mode == "RAG Mode":
                choices = ["Semantic", "Hybrid"]
            
                return ui.div(
                    ui.input_radio_buttons(
                        "method",
                        "RAG Retrieval Method",
                        choices=choices,
                        selected="Hybrid"
                    ),
                    class_="indented-group"
                )
            
            return None

    
    @output
    @render.ui
    def results_ui():
        res = search_results()
        if not res:
            return ui.p("Enter a query to begin.", style="color: white;")

        output_elements = []
        seen_titles = set()  

    # --- RAG MODE ---
        if res["mode"] == "RAG Mode" and res["answer"]:
            ai_answer = res["answer"]
            output_elements.append(
                ui.card(
                    ui.card_header(
                        ui.tags.strong("AI Analysis"),
                        style="font-size: 1.2rem; font-weight: 700; background-color: #f8f9fa;"
                    ),
                    ui.div(
                        ui.markdown(ai_answer),
                        style="font-size: 1.1rem; line-height: 1.6; color: #333; padding: 10px;"
                    ), 
                    style="margin-bottom: 20px; border: 2px solid #007bff; background: white;"
                )
            )
        
            output_elements.append(ui.tags.h3("Referenced Products", style="color: white;"))

            raw_docs = res["docs"] or []
            items_to_show = []
        
            for doc in raw_docs:
                title = doc.metadata.get("product_title", "")
                clean_title = title.strip().lower()
            
                if clean_title in seen_titles:
                    continue

                short_title = title[:20].lower()
                is_mentioned = short_title in ai_answer.lower()
            
                if is_mentioned:
                    items_to_show.append(doc)
                    seen_titles.add(clean_title)

            if len(items_to_show) < 3:
                for doc in raw_docs:
                    t = doc.metadata.get("product_title", "").strip().lower()
                    if t not in seen_titles:
                        items_to_show.append(doc)
                        seen_titles.add(t)
                    if len(items_to_show) >= 3: break

            for item in items_to_show:
                title = item.metadata.get("product_title", "Product Details")
                average_rating = item.metadata.get("average_rating")
                review_text = item.page_content

                output_elements.append(
                    ui.card(
                        ui.h5(title, style="font-weight: bold;"),
                        ui.p(
                            "Average Rating: ", rating_to_stars(average_rating),
                            style="margin-top: -5px; margin-bottom: 5px;"
                        ),
                        ui.p(
                            truncate_text(review_text, 150), 
                            style="font-style: italic; border-top: 1px solid #eee; padding-top: 10px; font-size: 1.1em;"
                        ),
                        style="margin-bottom: 15px; background: white; border-left: 5px solid #28a745;"
                    )
                )

    # --- SEARCH ONLY MODE ---
        elif res["mode"] == "Search Only":
            count = 0
            current_method = res.get("method", "")
            search_items = res.get("results") or []

            for item in search_items:
                product_name = item.get("product_title", "Product")
                title_key = item.get("product_title", "").strip().lower()

                if title_key not in seen_titles:
                    count += 1
                    display_rank = count

                    average_rating = item.get("average_rating")
                    rating_number = item.get("rating_number")
                    price = item.get("price")
                    score = item.get("score")

                    output_elements.append(
                       ui.card(
                        ui.div(
                            ui.span(
                                f"Rank {display_rank}", 
                                style="font-weight: bold; font-size: 1.2rem;"
                            ),
                            ui.span(
                                f"{current_method} score: {score:.3f}" if score is not None else "N/A",
                                style="font-weight: bold;"
                            ) if current_method != "Hybrid" else None,
                            style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;"
                        ),
                        
                        ui.h5(product_name, style="font-weight: bold;"),
                        
                        ui.p(
                            f"Average Rating: ", rating_to_stars(average_rating),
                            f" | Number of ratings: {display_text(rating_number)} | "
                            f"Price: {display_text(price)}",
                            style="margin-top: -5px; margin-bottom: -5px; font-size: 1.1rem;"
                        ),
                        
                        ui.p(
                            truncate_text(item.get("review_text", ""), 200),
                            style="font-style: italic; border-top: 1px solid #eee; padding-top: 10px; font-size: 1.1rem;"
                        ),
                        
                        style="margin-bottom: 15px; background: white;"
                    )
                )
                    seen_titles.add(title_key)

                if count >= 5: 
                    break

            if count == 0:
                return ui.p("No results found for this query.", style="color: white; font-weight: bold;")
            
        return ui.TagList(*output_elements)


app = App(app_ui, server, static_assets=Path(__file__).parent / "www")