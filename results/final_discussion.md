# Final Discussion

## Step 1: Improve Your Workflow

### Dataset Scaling

We examined our parquet dataset (100,000 sampled rows from merged product metadata and reviews) to verify that it contains at least 10,000 unique products. The original unique identifier (`parent_asin`) was dropped after merging, so we used `product_title` as a proxy for product identity. We found that there are 56,852 unique product titles in our sampled dataset. However, we note that `product_title` is not a guaranteed unique identifier, since the same product could be titled differently, so we validated this assumption with two tests.

First, we manually inspected a random sample of 100 unique product titles and found them to be observably distinct products. Then, using semantic similarity, we retrieved the most similar products for a sample of 10 product titles and examined the results. We found that exact matches correspond to the same product, while highly similar titles typically represent distinct variants (e.g., different brands, sizes, or materials), rather than duplicates.

Together, these checks suggest that `product_title` provides a reasonable approximation of product-level uniqueness. Therefore, we conclude that our dataset contains well over 10,000 unique products. Code for this validation is provided in the `milestone3_exploration` notebook.

### LLM Experiment

#### Models Compared

We compared two LLMs with different sizes and capabilities:

- **Model 1:** `llama-3.1-8b-instant`  
  - Family: LLaMA 3.1  
  - Size: 8 billion parameters  
  - Characteristics: Fast, lightweight, and generally strong at following structured prompt instructions.

- **Model 2:** `llama-3.3-70b-versatile`  
  - Family: LLaMA 3.3  
  - Size: 70 billion parameters  
  - Characteristics: Much larger model with stronger reasoning and generation capabilities, expected to better handle complex queries and synthesize information across retrieved documents.

---

#### Prompt Used

To ensure a fair comparison, both models were evaluated using the same prompt and identical retrieved context (semantic retriever). The following prompt was used for initial comparison and was later refined (see final prompt below).

```python
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
```

#### Results

We evaluated both models on five representative queries ranging from simple keyword-based queries to more complex, intent-based queries. Full outputs for both models are shown below for transparency, using identical retrieved context for each query. Additionally, for reproducibility the code run to generate this output is in the `milestone2_exploration.ipynb` notebook. 


##### Outputs for both models on 5 Queries 

```python
"""
====================================================================================================
QUERY 1: 6-inch ceramic plant pot with drainage hole
====================================================================================================

llama-3.1-8b-instant
--------------------
Based on your request, I've found several options that fit your criteria. Here are some gift ideas for a 6-inch ceramic plant pot with a drainage hole:

1. **ADN Ceramic Planter, 6 Inch Plant Pots for Plants Modern Decor Indoor Flower Pot with Saucer and Drainage Hole, White** ($24.99) - This modern and simple design ceramic planter is a great choice. It has a drainage hole and a saucer to catch any excess water. Reviewers love its stylish look and functionality.
2. **YFFSRJDJ Ceramic Flower Plants Pots Planter with Drainage Hole, 6.0 inch+5.0 Inch. Indoor-Outdoor Large Round Succulent Orchid Pot Set (Blue)** ($26.99) - These gorgeous planters have a modern design and a drainage hole. They're perfect for indoor or outdoor use and come in a beautiful blue color. Reviewers rave about their beauty and value.
3. **Gohearin Ceramic Plant Flowerpot, Hand-Painted Flowerpot Set, 6.6+5.4-inch Indoor Flowerpot with Drain Hole, Used for Indoor Plants, Cactus, Succulent Plants and Other Outdoor Garden, 2-Piece Set** ($29.99) - This hand-painted flowerpot set has a unique design and a drainage hole. It's perfect for indoor or outdoor use and comes in a beautiful set of two. Reviewers love its display value and functionality.

All of these options are available in a 6-inch size and have a drainage hole, making them perfect for your needs.

llama-3.3-70b-versatile
-----------------------
For a 6-inch ceramic plant pot with a drainage hole, consider the ADN Ceramic Planter, 6 Inch Plant Pots for Plants Modern Decor Indoor Flower Pot with Saucer and Drainage Hole, White, priced at $24.99, which features a modern simple design and is suitable for indoor decoration. 

Another option is the Gohearin Ceramic Plant Flowerpot, Hand-Painted Flowerpot Set, which includes two pots, one of which is close to 6 inches, and is priced at $29.99. This set has a unique hand-painted design and is suitable for indoor and outdoor use.

You may also consider the YFFSRJDJ Ceramic Flower Plants Pots Planter with Drainage Hole, 6.0 inch+5.0 Inch, priced at $26.99, which features a modern design and comes in a set of two pots. This planter is ideal for succulents, orchids, and other plants, and its beautiful design makes it a great addition to any room. 

Additionally, the DEARDESIRES 10 inch Ceramic Plant Pot is a larger option that also features drainage holes, although it's a bit bigger than the requested 6-inch size. 

Lastly, if you're looking for a smaller set of planters, the Ufrount 3 Inches Ceramic Planter Pot with Drainage Holes is a set of 6 small pots that can be used for mini plants, succulents, and herbs.


====================================================================================================
QUERY 2: blue patio chair cushion
====================================================================================================

llama-3.1-8b-instant
--------------------
Based on your search for a blue patio chair cushion, I recommend the following three products:

1. **Rocking Chair Cushion Pad, 47IN Indoor Outdoor Navy Blue** ($39.99) - Although this product received a low rating in one review, it's worth considering due to its durability and softness. However, please note that the reviewer experienced fading issues.

2. **LVTXIII Outdoor Seat Cushions Patio Chair Pads with Ties, All Weather Chair Cushions for Home Office and Patio Garden Furniture Decoration D16”xW17”, Stripe Beige, Set of 2** ($39.99) - This product is not exactly blue, but beige with a stripe pattern. However, it's a good option for those looking for a durable and fade-resistant cushion. You could consider customizing the cushion or using a cover to achieve a blue hue.

3. **Indoor/Outdoor Bench Cushion Cotton Garden Furniture Loveseat Cushion, 51.2"x19.7" Patio Wicker Seat Cushions for Lounger Garden Furniture Patio Lounger Bench (Navy Blue)** ($34.99) - This product is a good option if you're looking for a navy blue cushion. However, please note that it's made of indoor material and may not be suitable for heavy outdoor use.

Please keep in mind that these recommendations are based on the provided context, and you should research each product further to ensure it meets your specific needs.

llama-3.3-70b-versatile
-----------------------
For a blue patio chair cushion, consider the Rocking Chair Cushion Pad, 47IN Indoor Outdoor Navy Blue, although it had a disappointing review, it's one of the few blue options available. Alternatively, you could look at the Pillow Perfect Stripe Indoor/Outdoor Chair Seat Cushion, which comes in a Red/White color but has a high rating and is weather and fade resistant, priced at $82.99. Another option is the LVTXIII Outdoor Seat Cushions, which come in a Stripe Beige color and are priced at $39.99, they have a high average rating and are fade-resistant. The Indoor/Outdoor Bench Cushion Cotton Garden Furniture Loveseat Cushion is also available in Navy Blue, priced at $34.99, but had a review stating it's not suitable for outdoor use. Lastly, the Indoor/Outdoor Bench Cushion, Swing Cushion, is available in Gray, priced at $29.99, but had a review stating it's too short and thin.


====================================================================================================
QUERY 3: something to make a garden look more colorful and lively
====================================================================================================

llama-3.1-8b-instant
--------------------
Based on the provided reviews, here are three gift ideas to make a garden look more colorful and lively:

1. **Evergreen Garden Gazing Ball in Purple Iridescent Design Everyday Decor for Homes Gardens and Yards** ($59.94) - This beautifully designed gazing ball adds stunning springtime details to your lawn and yard. Its unique iridescent leaves in a purple pattern create a gorgeous and eye-catching effect.
2. **Life In Color Black Gnome Garden/Planter Decor (Rainbow)** - This colorful gnome is a great addition to any garden, adding a playful and whimsical touch. Its vibrant colors and cute design make it a perfect gift for anyone looking to add some personality to their outdoor space.
3. (Alternative option, not listed as the first due to lower rating) **Exhart Garden Solar Lights, Set of 2 Decorative Bubble Garden Stakes, Color Changing LED, Outdoor Garden and Yard Decoration, 2 x 30 Inch, Square** ($23.99) - These colorful solar lights feature color-changing LED lights decorated with suspended bubbles, adding a touch of enchanting beauty to any outdoor garden or yard.

These options are great for making a garden look more colorful and lively, with the gazing ball and gnome adding a beautiful and unique touch, while the solar lights provide a fun and interactive element.

llama-3.3-70b-versatile
-----------------------
Consider the Evergreen Garden Gazing Ball in Purple Iridescent Design Everyday Decor for Homes Gardens and Yards, priced at $59.94, which features a beautiful gazing ball with iridescent leaves in a purple pattern, adding stunning springtime details to your lawn. 

The Life In Color Black Gnome Garden/Planter Decor (Rainbow) is another option, made of durable resin material with vibrant colors, perfect for adding a pop of color to your garden. 

For a unique touch, the Little Garden Gnome 300 Pcs Glow in The Dark Pebbles for Walkways and Decor can be used to create a colorful pathway, although it's essential to note that some customers have reported issues with the glow-in-the-dark feature. 

Alternatively, you could opt for the Exhart Garden Solar Lights, Set of 2 Decorative Bubble Garden Stakes, priced at $23.99, which feature color-changing LED lights and suspended bubbles, adding an enchanting beauty to your outdoor garden, despite some customers finding them not very bright.


====================================================================================================
QUERY 4: something to keep the deer out of my garden without hurting them
====================================================================================================

llama-3.1-8b-instant
--------------------
Based on the reviews, here are three gift options to keep deer out of your garden without harming them:

1. **Deer Repel Deer Repellent Plants Pouches** ($34.97) - This product uses natural ingredients and can be a convenient way to protect your garden. Although it didn't work for the reviewer in a suburban area with aggressive deer, it may still be effective in other areas or for smaller deer populations.
2. **Easy Gardener 6050AZ Netting and Fencing** ($24.9) - This netting and fencing product is designed to keep deer and other animals out of your garden. Reviewers have found it to be effective in blocking birds and other pests, and it can be reused. It may require some effort to set up around your garden.
3. **Deer Out 40oz Ready-to-Use Deer Repellent** ($17.95) - This product has been found to be effective in keeping deer away from plants and has a minty fresh scent that won't harm them. Reviewers have reported that it works well for a variety of plants, including tomatoes and hostas.

llama-3.3-70b-versatile
-----------------------
Consider the Deer Repel Deer Repellent Plants Pouches, priced at $34.97, which uses all-natural ingredients to keep deer and rabbits away from your garden. Alternatively, the Easy Gardener 6050AZ Netting and Fencing, priced at $24.9, is a physical barrier that can protect your plants from deer and other animals. Another option is the Deer Out 40oz Ready-to-Use Deer Repellent, priced at $17.95, which has a minty fresh scent that repels deer without harming them. The Deer Out repellent has been effective for many users, including one who found it to be the only solution that kept deer from eating their tomato plants and hostas. The Easy Gardener netting is also reusable and lightweight, making it a practical choice.


====================================================================================================
QUERY 5: what is a good plant to keep outside that won’t die if I forget to water it and doesn’t need much sunlight
====================================================================================================

llama-3.1-8b-instant
--------------------
Based on the provided Amazon product reviews and metadata, here are three gift options that can thrive in outdoor settings with minimal watering and sunlight:

1. **Bloomify Live Sundew Plant Terrarium**: This enclosed ecosystem is perfect for someone who wants a low-maintenance plant. The terrarium is self-sustaining and requires zero maintenance. It's a unique gift for any home décor enthusiast. ($29.99)
2. **Bloomify Live Air Plant Terrarium**: Although not specifically mentioned in the provided documents, Bloomify is a brand that offers low-maintenance plants, and their Air Plant Terrarium is a great option. It doesn't require soil or direct sunlight, making it perfect for someone who tends to forget to water their plants. ($34.99)
3. **Sundew Plant in a Glass Container**: While not explicitly mentioned, Sundew plants are often used in low-maintenance terrariums. They can thrive in low-light conditions and require minimal watering. This specific product is not available, but you can consider similar products from other brands.

llama-3.3-70b-versatile
-----------------------
Considering your requirements, I'd recommend the following plants:

1. Bloomify Live Sundew Plant Terrarium (price not available) - This plant is a great option as it's described as a "maintenance-free" and "self-sustaining" terrarium, which implies it can thrive with minimal care and watering. It's also suitable for indoor use, but can be kept outside in shady areas.
2. Unfortunately, I couldn't find other plant options that fit your criteria. The Luffy Coco Philippines Java Fern has received multiple reviews stating it arrived dead, which doesn't inspire confidence in its hardiness or ability to survive with minimal care.
3. Since I don't have more information on other plants, I would suggest exploring other products on Amazon that specifically mention being "drought-tolerant" or "low-maintenance" to find a plant that fits your needs.
"""
```

#### Discussion and Key Observations

The outputs from the two models show clear structural differences. The baseline model `llama-3.1-8b-instant` consistently produces structured responses, including an opening statement followed by a numbered list of products with bolded titles and a concluding statement. In contrast, the larger model `llama-3.3-70b-versatile` produces responses that are more conversational, with product recommendations embedded within paragraphs rather than listed formally.

Additionally, the models differ in their reasoning. The simpler baseline model `llama-3.1-8b-instant` appears less capable of fully understanding the intent of the prompt and instead focuses on rigidly following its structure. For example, the prompt indicates that the LLM ought to avoid generic phrases like "this is a great gift" without adding a specific reasoning, but it ends up framing 4 out of 5 of the product responses as "gift recommendations" that are not queried as gift recommendations. It also produces slightly awkward phrasing in places, suggesting it is attempting to follow prompt instructions but not fully integrating them. For instance, statements such as “Please keep in mind that these recommendations are based on the provided context…” appear to be an attempt to satisfy the instruction to acknowledge uncertainty.

In contrast, the larger model `llama-3.3-70b-versatile` incorporates prompt instructions in a more natural and fluent way. Its responses are easier to read, better integrated, and more conversational, while still including relevant product details such as price and review-based reasoning.

The models also differ in how they handle incomplete context. The smaller model sometimes includes less relevant products in order to satisfy the requirement of providing at least three recommendations. In contrast, the larger model is better able to recognize when the available context does not fully satisfy the query and explicitly acknowledges these limitations, rather than generating unsupported recommendations. This is particularly evident in Query 5. However, even the larger model occasionally maintains a numbered structure, likely due to the prompt requirement to include at least three recommendations.

Overall, these differences are most pronounced for more complex queries. The 8B baseline model prioritizes consistency and structure, while the 70B model prioritizes contextual accuracy and relevance, even when this results in less strictly formatted outputs.

#### Performance Considerations

We observed that the 70B model has slightly slower response times (1.2 seconds) compared to the 8B model (0.9 seconds). This highlights a tradeoff between response quality and speed, where larger models provide improved reasoning at the cost of slower inference. However, the difference is not significant enough in this case to outweigh the benefits of the more complex model.


#### Model Selection

Overall, the `llama-3.3-70b-versatile` model produced higher-quality outputs, with stronger reasoning, better handling of incomplete context, and more natural explanations. While the `llama-3.1-8b-instant` model demonstrated faster and more consistent formatting, its tendency to prioritize structure over contextual accuracy resulted in weaker recommendations for more complex queries.

Based on this comparison, we selected the 70B model as the default for our RAG pipeline, as it provides a better balance of accuracy, reasoning, and user-facing response quality. However, we did prefer the structured format of the simpler model’s output for readability, as it is easier to process recommendations when products are clearly separated and highlighted. As a result, we will use the more complex model while refining the prompt to improve output formatting.

#### Final Prompt Optimized for Selected Model

We refined our final prompt to balance structure, relevance, and usability in an application setting. The final version enforces a clear, numbered format while prioritizing strong matches to the query and avoiding filler or unsupported recommendations. We also guide the model to use review insights for explanation rather than repeating rating statistics, resulting in more natural and informative outputs. These changes improved consistency and reduced irrelevant results, making the responses better suited for a product search interface. We also relaxed the earlier requirement to always return at least three recommendations, as we found that enforcing this constraint often introduced irrelevant or weak matches when the retrieved context was limited.

```python
SYSTEM_PROMPT_OPTIMIZED = """
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
```

## Step 2: Additional Feature (Option 3: Scale to $\geq$ 100k Products)

### What We Implemented

We scaled our dataset by increasing the total number of unique products from 56,852 to 101,268. To achieve this, we increased our sample size to 225,000 rows (up from the original 100,000 rows) from the source review and metadata dataset. To ensure the final dataset reached our goal of over 100,000 unique products, we validated the count using parent_asin, the unique identifier, rather than relying solely on product titles, which may be missing or incomplete.

We improved the quality of the dataset and the knowledge base for the overall retrieval pipeline by replacing the previous parquet file with a new parquet file that is aggregated at the product level and has multiple reviews per product row. Our final parquet file is similar in size to the original file but contains more information, because it has 1.78 times more products and multiple reviews per product - thus maintaining a similar file size while representing more products and richer product-level information. 

```python
review_title      142083
text              211492
parent_asin       101268
product_title     100647
average_rating        41
rating_number       5561
features           86453
description        52345
price               9570
categories           756
dtype: int64
```

### Key Changes

- Aggregated review-level data into product-level documents, combining multiple reviews into a single `review_text` field per product, and saved as a parquet file
- Updated both semantic and BM25 pipelines to use the scaled dataset (`processed_scaled_sample.parquet`)
- Incorporated TA feedback and refactored the BM25 pipeline to save the tokenized corpus and metadata instead of a serialized retriever, improving reproducibility and making the pipeline more robust and maintainable at scale by separating preprocessing from retrieval construction
- Rebuilt both the FAISS vector store and BM25 corpus and metadata using the scaled dataset to enable efficient retrieval over a significantly larger set of product-level documents

### Key Results

- Reduced redundancy in retrieval results by eliminating duplicate products appearing multiple times
- Improved relevance of retrieved documents, especially for semantic and hybrid search
- Maintained reasonable build, run, and query performance despite increased dataset size

### Reference Code
```python
# Load the review data
reviews_df = pd.read_json(main_file, lines=True)
# Load the meta data
meta_df = pd.read_json(meta_file, lines=True)
# Drop columns
reviews_df.drop(
    columns=['images', 'asin', 'user_id', 'rating',
             'timestamp', 'helpful_vote', 'verified_purchase'
             ],
    inplace=True
)
meta_df.drop(
    columns=['main_category', 'images', 
             'videos', 'store', 'details',
             'bought_together', 'subtitle', 
             'author'
             ],
    inplace=True
)
# Rename columns
reviews_df.rename(
    columns={
        "title": "review_title"
    },
    inplace=True
)

meta_df.rename(
    columns={
        "title": "product_title"
    },
    inplace=True
)
# Convert columns with lists of text to simple strings rather than lists 
def flatten_to_text(x):
    if isinstance(x, list):
        return " ".join(str(item) for item in x if pd.notna(item))
    if pd.isna(x):
        return ""
    return str(x)

for col in ["features", "description", "categories"]:
    meta_df[col] = meta_df[col].apply(flatten_to_text)

# Check Result to ensure it worked properly
meta_df[["features", "description", "categories"]].head()
# Merge the review data and metadata using "parent_asin"
merged_df = pd.merge(
    reviews_df, 
    meta_df, 
    on='parent_asin', 
    how='inner'
)

# Sample rows from the fully merged dataframe 
scaled_sample = merged_df.sample(n=225000, random_state=42)

# Checking whether the scaled sample has more than 100k products
scaled_sample.nunique()

# Aggregate reviews at the product level using parent_asin
scaled_sample = (
    scaled_sample
    .groupby("parent_asin", as_index=False)
    .agg({
        "review_title": "first",
        "product_title": "first",
        "average_rating": "first",
        "rating_number": "first",
        "features": "first",
        "description": "first",
        "price": "first",
        "categories": "first",
        "text": list
    })
)

# Combine list of review texts into a single string per product
scaled_sample["review_text"] = scaled_sample["text"].apply(
    lambda x: " ".join([str(i) for i in x if isinstance(i, str)])
)

# Drop raw text and parent_asin 
scaled_sample = scaled_sample.drop(columns=['text', 'parent_asin'])

# Optimize data types to reduce memory usage and improve processing efficiency
scaled_sample["rating_number"] = pd.to_numeric(scaled_sample["rating_number"], downcast="integer")
scaled_sample["average_rating"] = pd.to_numeric(scaled_sample["average_rating"], downcast="float")
scaled_sample["categories"] = scaled_sample["categories"].astype("category")

# Clean and extract numeric values from price column
scaled_sample["price"] = (
    scaled_sample["price"]
    .replace(["—", "–", "", "N/A", "na", "null"], np.nan)
    .astype(str)
    .str.extract(r"(\d+\.?\d*)")[0]
)

# Save scaled sample dataset as a compressed Parquet file
output_path = OUT_DIR / "processed_scaled_sample.parquet"
scaled_sample.to_parquet(output_path, index=False, compression="snappy")
```
  
## Step 3: Improve Documentation and Code Quality

### Documentation Update

- READ ME Updates: 
    - Updated Data Processing section to reflect aggregation from review-level to product-level documents, where each row represents a single product with multiple reviews combined into a `review_text` field
    - Updated dataset description to reflect scaling to 100,647 product-level documents and clarified sampling after aggregation
    - Updated BM25 build instructions to reflect the new corpus and metadata storage 
    - Updated Semantic Search explanation to reflect updated semantic scoring function with normalized embeddings and cosine similarity
    - Updated Retrieval Methods section to ensure descriptions accurately reflect implemented pipelines
    - Updated LLM model and prompt description to reflect the change from `llama-3.1-8b-instant` to `llama-3.3-70b-versatile` and the optimized prompt (as discussed above in section LLM Experiment)
    - Updated RAG flow diagram to reflect 50/50 hybrid setup with BM25 and semantic search
    - Updated build instructions to clarify reuse of saved BM25 and FAISS artifacts
    - Added and refined Notes section to document known limitations of the RAG pipeline
    - Improved Reproducibility and Setup section for clarity and ease of use (eg. added note about Semantic vector store build time)
    - Added Milestone 3 to Project overview, Features, and Evaluation sections for consistency


### Code Quality Changes

We made several code quality improvements to make the repository more reproducible, maintainable, and consistent:

- removed hardcoded file paths and replaced them with shared configuration variables using `pathlib.Path`
- centralized key project constants such as data paths and model names in `src/config.py`
- updated semantic search to use cosine similarity with normalized embeddings, addressing TA feedback and improving the consistency and interpretability of similarity scores
- removed unused variables and outdated code
- added or updated function docstrings across the codebase
- refactored the BM25 pipeline to store a tokenized corpus and metadata instead of a serialized retriever, addressing TA feedback and improving reproducibility
- updated the semantic pipeline to suppress unnecessary Hugging Face warnings, improving the user experience
- updated the hybrid retrieval setup to use balanced (50/50) weighting between BM25 and semantic results
- added and refined `.gitignore` to exclude large artifacts (e.g., FAISS store, BM25 corpus) and environment files, improving repository cleanliness and reproducibility
- Updated the app.py code so that the presentation of price does not show up as "nan" but instead consistently shows "N/A".  And adjusted the presentation of referened products in RAG mode so that the review text doesn't include the Product title at the beginning. 


## Step 4: Cloud Deployment Plan

Below we summarize our plan for how we would deploy our application with cloud computing.

### Data Storage 
- Raw data: Stored in AWS S3 as the central, durable, and low-cost storage layer for original datasets.
- Processed data: The aggregated product-level parquet file is also stored in S3 under separate prefixes, allowing for versioning and easier updates as new data is incorporated. S3 is chosen because it provides durable, scalable, and cost-efficient storage for large datasets and retrieval index files.
- Vector index: Stored in S3 (justified above) as a FAISS store directory containing the FAISS index file. It will be loaded into EC2 memory at startup for low-latency semantic search.
- BM25 index: The tokenized corpus and associated metadata are also stored in S3 (justified above), in a BM25 store directory, and are loaded into memory on EC2 during startup for fast keyword retrieval.

### Compute
The application is built using Shiny and runs on a cloud-hosted AWS EC2 instance, which serves the Shiny web interface and executes the full retrieval and RAG pipeline.  Compared to AWS Elastic Beanstalk, EC2 is preferred because it provides greater flexibility and customization, which is important for a Shiny-based application with custom machine learning components.

The EC2 instance hosts the BM25, semantic search, and hybrid retrieval methods, as well as LLM API calls for the RAG system. To ensure efficient performance under concurrent usage, the BM25 index and semantic vector store are stored in AWS S3 and loaded into memory once when the EC2 instance starts. This avoids repeated loading or recomputation for each request and improves response speed. If traffic increases in the future, the system can be scaled to handle higher concurrency by increasing API quotas. In addition, multiple EC2 application instances could be placed behind an Application Load Balancer to distribute traffic and improve availability. This setup can also be combined with Auto Scaling for more automated scaling.

For LLM inference, we use an API-based approach (e.g., Groq) rather than a self-hosted model. This avoids the need to deploy and maintain large language models, significantly reducing infrastructure cost and technical complexity (e.g., GPU requirements). Rate limiting is governed by the LLM API provider’s free-tier usage limits to control cost. Each user query is sent to the LLM API along with retrieved context from BM25 and semantic search. The response is then returned to the Shiny app for display.

### Streaming/Updates
To incorporate new products and reviews, and keep the system up to date, we use an automated data pipeline that runs on a regular batch schedule (e.g., weekly). On each run, the pipeline ingests newly available product and review data from external sources, stores the raw data in S3, preprocesses the data (including cleaning, feature engineering, and aggregation at the product level), and regenerates the BM25 corpus and metadata as well as the semantic vector store (FAISS). The updated indices are then saved back to S3 to ensure consistency across the system.

In this design, indices are updated via periodic batch re-indexing rather than real-time updates. This approach simplifies the system and avoids the high computational cost of rebuilding indices for each individual update while ensuring consistency of retrieval results.

To prevent these more computationally intensive steps from affecting application performance, the preprocessing and re-indexing pipeline can be run on a separate EC2 instance. This instance processes newly ingested data, rebuilds the retrieval indices, and can be shut down after completion, while the main application instance continues to serve user requests.