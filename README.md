# Smart Amazon Product Query Assistant

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

We use a subset of the Amazon product review dataset (Patio, Lawn & Garden category).

The dataset consists of two sources:

- **Reviews data**
  - `review_title`
  - `text` (review content)
  - `rating`
  - `parent_asin` (used for joining)

- **Metadata data**
  - `product_title`
  - `description`
  - `features`
  - `categories`
  - `average_rating`
  - `rating_number`
  - `price`
  - `parent_asin` (used for joining)

---

## Data Processing

The raw review and metadata files are merged using `parent_asin`.

We retain a subset of fields relevant for retrieval and display.

For semantic search, we construct a combined text field:

- product title  
- review title  
- review text  
- features  
- description  
- categories  

Basic preprocessing includes:
- handling missing values  
- converting fields to strings  
- concatenating text fields into a single document per row  

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

## Retrieval Methods

### BM25 Retrieval

- Uses a tokenized version of the dataset
- Applies simple preprocessing:
  - lowercasing  
  - punctuation removal  
  - whitespace tokenization  
- Retrieves documents based on keyword matching and term frequency

### Semantic Search

- Uses `sentence-transformers/all-MiniLM-L6-v2`
- Documents are embedded into vector representations
- FAISS is used to build an index for efficient similarity search
- Queries are encoded and matched against document embeddings
- Results are ranked based on vector similarity

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

Build Semantic Index

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

- The FAISS index file is not included in the repository due to size constraints and is generated locally. 
- Duplicate results may occur because each review is treated as a separate document rather than aggregating at the product level. 

These limitations highlight opportunities for improvement in later milestones, including aggregation at the product level and hybrid retrieval approaches.