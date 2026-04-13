# Amazon Patio, Lawn and Garden Product Search

Team Members: Sasha S, Claire Saunders 

## Project Overview

This project builds a context-aware product search assistant that retrieves relevant Amazon products based on natural language queries. This work was developed as part of the DSCI 575: Advanced Machine Learning course.

In Milestone 1, we focus on retrieval only (no LLMs), implementing and evaluating two approaches:

- **BM25 (keyword-based retrieval)**
- **Semantic search using sentence embeddings and FAISS**

These two approaches allow us to compare traditional keyword-based retrieval with modern embedding-based methods. The system returns the most relevant products for a given query and is designed to support a user-facing web application built with Shiny.

---

## Features (Milestone 1)

- Semantic search using SentenceTransformers and FAISS
- BM25 keyword-based retrieval
- Query evaluation across different levels of complexity
- Structured output for easy integration into a web app

---

## Dataset

We use a subset of the Amazon Reviews 2023 dataset, specifically the Patio, Lawn & Garden category.

The dataset consists of two sources:

- **Reviews data**, containing user-generated content such as review text, titles, and ratings  
- **Metadata data**, containing product-level information such as product title, description, features, categories, price, and aggregate ratings  

These two sources are linked using the `parent_asin` identifier.

The dataset is sourced from:

- https://amazon-reviews-2023.github.io/  
- https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023  

---

## Data Processing

The raw review and metadata files are merged using `parent_asin`.

We retain a subset of fields relevant for retrieval and display.

For both BM25 and semantic search, we construct a combined text field by concatenating:

- product title  
- review title  
- review text  
- features  
- description  
- categories  

In addition to this combined text used for retrieval, the following metadata fields are retained to support result display in the web application:

- `product_title`  
- `review_title`  
- `review_text`  
- `rating`  
- `average_rating`  
- `rating_number`  
- `price`  

Basic preprocessing includes:
- handling missing values (e.g., price and sparse metadata fields)  
- ensuring all text fields are consistently formatted as strings  
- converting list-based fields (features, description, categories) into plain text  
- concatenating text fields into a single document per row  

Additional preprocessing is applied depending on the retrieval method:

- **BM25:** tokenization (lowercasing, punctuation removal, whitespace splitting)  
- **Semantic search:** uses the combined text directly and encodes it into embeddings  

---

### Data Sampling and Size Considerations

Due to the large size of the fully merged dataset (~14GB), we use a subset of 100,000 rows for this project.

This subset was created by randomly sampling from the fully merged dataset after combining the reviews and metadata tables. While this approach does not guarantee preservation of all underlying distributions, it is expected to retain a representative mix of products, reviews, and categories without introducing systematic bias.

This design choice allows the project to:
- remain within GitHub file size limits
- be reproducible without requiring large external storage
- run locally on machines with typical RAM constraints

The processed dataset is stored as a parquet file and included in the repository to support easy setup and reproducibility. While the reduced dataset may limit full coverage of the product space, it is sufficient for evaluating retrieval performance across a range of query types.

---

# Retrieval Methods

### BM25 Retrieval

- Uses a tokenized representation of the dataset  
- Performs keyword-based retrieval using term frequency and inverse document frequency  
- Ranks documents based on how well the words in the query match the words in the document  
- Returns a BM25 relevance score for each result, where higher scores indicate better matches  

### Semantic Search

- Uses `sentence-transformers/all-MiniLM-L6-v2` to encode text into dense vector embeddings  
- FAISS is used to build an index for efficient similarity search 
- Queries are encoded and used to retrieve the most similar documents from the FAISS index
- Retrieval is based on embedding distance (closer = more similar)  

For interpretability, distances are converted into a similarity-style score:

```python
score = 1 / (1 + distance)
```

so that higher scores correspond to closer matches.

---

## Reproducibility and Setup

### 1. Clone the repository

```bash
git clone https://github.com/UBC-MDS/DSCI_575_project_saunde95_sashasph.git
```
If you have SSH configured, you may use the SSH URL instead of HTTPS.

### 2. Create and Activate Environment

Ensure you are in the project root, then create the environment from the yml file.

```bash
cd DSCI_575_project_saunde95_sashasph/
conda env create -f environment.yml
conda activate retrieval-app
```

### 3. Build Retrieval Artifacts (Required Before Running the App)

Both retrieval methods require a one-time local build step.

Build BM25 Retriever:
```bash
python -m src.build_bm25
```

This will: 
- create tokenized documents
- build the BM25 retriever
- save them to `data/processed/` as `.pkl` files

Build Semantic Index:

```bash
python -m src.build_semantic_index
```

This will: 
- generate embeddings
- build a FAISS index
- save it to `data/processed/faiss_index.index` 

These steps only need to be run once. If the saved files already exist, they will be reused.

### 4. Run the app locally

```bash
shiny run app/app.py
```
Then open the provided local URL in your browser. 

---

## Evaluation

We created a set of 10 queries with varying levels of complexity: 
- keyword-based queries
- semantic (natural language) queries
- complex multi-constraint queries

Both BM25 and semantic retrieval methods were evaluated by retrieving the top 5 results for each query. 

A subset of 5 queries was selected for detailed comparison. 

Results and discussion can be found in: 
- results/milestone1_discussion.md
- notebooks/milestone1_evaluation.ipynb 

---

## Notes

- Retrieval artifacts for both BM25 (tokenized documents and retriever object) and semantic search (FAISS index) are generated locally and stored in `data/processed/`. These files are not included in the repository due to size constraints.  
- Duplicate results may occur because each review is treated as a separate document rather than aggregating at the product level.  

These limitations highlight opportunities for improvement in later milestones, including aggregation at the product level.