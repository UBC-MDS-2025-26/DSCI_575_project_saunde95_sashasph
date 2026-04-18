# Milestone 2: Qualitative Hybrid RAG Evaluation

## Model Choice

We selected `llama-3.1-8b-instant` via the Groq API for our RAG pipeline. Compared to smaller models (e.g., 0.8B–4B), this 8B-scale model provides stronger instruction-following and more coherent responses, which is important for synthesizing multiple retrieved documents into useful product recommendations.

We chose this model size because it offers a good balance between quality and efficiency while still being free to use through the Groq API. Smaller models would likely be faster and lighter, but we expected them to be less reliable for combining context into grounded recommendations. Using the Groq API allowed us to access a higher-capability model without requiring local GPU resources, while still maintaining fast enough inference for an interactive application.

## Prompt Design

To design an effective RAG system for product recommendations, we experimented with multiple system prompt variants and evaluated their impact on response quality.

### Prompt Iterations

**Prompt 1 (Baseline):**
```text
You are a helpful Amazon shopping assistant.

Answer the user's question using the provided product reviews and metadata.
Provide helpful suggestions and explain your reasoning.
```

This initial prompt allowed flexible responses, but often resulted in outputs that were less grounded, occasionally vague, and inconsistent in structure.

**Prompt 2 (Grounding and Constraints):**

```text
You are a helpful Amazon shopping assistant.

Answer the user's question using ONLY the provided Amazon product review and metadata context.
Do not make up any information.

If the context is not sufficient to answer confidently, say that the available reviews do not provide enough information.

Keep the answer short, factual, and grounded in the retrieved documents.
```
This version improved factual accuracy and reduced hallucination by enforcing stricter grounding. However, the responses were still somewhat generic and lacked consistent structure for recommendations.

**Prompt 3 (Task-Specific Refinement):**
```text
You are a helpful Amazon shopping assistant.

Answer the user's question using ONLY the provided Amazon product review and metadata context.
Do not make up details that are not supported by the context.
If the context is not sufficient to answer confidently, say so.

When answering:
- keep the answer concise and practical
- recommend only products supported by the retrieved context
- mention product titles when helpful
- avoid repeating duplicate or near-duplicate products
- always give at least 3 recommendations, and include up to 5 when more good options exist

```
This version introduced task-specific constraints aligned with the goal of generating structured product recommendations. It improved consistency, reduced duplication, and ensured that responses included a sufficient number of relevant products but it also introduced minor issues such as explicitly stating when price was not provided, which was unnecessary for the user experience.

**Final Prompt (Refined for Application Use):**
```text
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
```

This final prompt builds on Prompt 3 by further refining the structure and tone of the responses. In particular, it improves:

- usability by including price and clearer product descriptions
- readability by enforcing natural explanations and removing unnecessary statistics
- consistency and improved user experience by preventing generic phrasing and redundant summaries
- output quality by encouraging more meaningful distinctions between recommendations

Overall, this version produced the most consistent, user-friendly, and practically useful outputs, and was used in the final RAG pipeline.

## Per-Query Evaluation

### Query 1: "6-inch ceramic plant pot with drainage hole"

**Hybrid Output:**  

Based on your search for a 6-inch ceramic plant pot with a drainage hole, I recommend the following options:

1. **D'vine Dev 6 Inch Ceramic Planter Pot**: This modern cylindrical planter pot is well-made, sturdy, and has excellent quality. It comes with a saucer and drainage hole, making it perfect for indoor plants. ($24.99)
2. **Small Plant Pots Set** (6-inch) from Document 4: This set of 2 plant pots includes a 6-inch diameter pot with a drainage hole and saucer. The speckled grey design is elegant and modern, making it a great choice for indoor decoration. ($26.99)
3. **ADN Ceramic Planter** (6-inch) from Document 1: Although the product title mentions a 6-inch pot, I couldn't confirm this size in the product features. However, it does have a drainage hole and a modern design that might be appealing to you. ($24.99)

Please note that I couldn't find a 6-inch ceramic plant pot with drainage hole in the other documents. If you'd like to consider other options or have further questions, feel free to ask!

**Accuracy:**  
Yes: The recommendations are factually correct and align well with the query. The products match the requested specifications, including size and drainage features.

**Completeness:**  
Yes: The response fully addresses the query by providing multiple options that meet all constraints.

**Fluency:**  
Yes: The response is clear, structured, and easy to read.

**Other Observations:**  
The output mentions that it could not find additional matching products in "other documents," which could be slightly confusing for users since that phrasing is internal to the system. However, this does not significantly impact the usefulness of the response, and the recommendations remain clear and helpful overall.

---

### Query 2: "blue patio chair cushion"

**Hybrid Output:**  
Based on your search for a blue patio chair cushion, I recommend the following options:

1. **REDCAMP Chaise Lounge Chair Cushion for Patio Furniture** ($39.99) - This cushion is thicker and more comfortable than some of the other options, with a zero-gravity design and anti-slip cover. Although it was considered too thin by one reviewer, it's worth considering if you prioritize comfort.

2. **Rocking Chair Cushion Pad, 2 Pieces Indoor/Outdoor Rocking Chair Cushions Set** ($29.99) - These cushions come in navy blue and are a good option if you're looking for a set of two. They're made of 100% polyester and have a soft, comfortable design.

3. There is currently no other option that is directly related to the search term blue patio chair cushion that is highly rated and available.

However, there is one other blue option that might be worth considering, the **Rocking Chair Cushion Pad, 47IN Indoor Outdoor Navy Blue Rocking Chair Cushions Set**, however, this product has a very low average rating (2.9) and has been described as very flimsy and poorly made.

**Accuracy:**  
Partial: The recommendations are only partially accurate. Some products match the intended category, but not all strictly meet the “blue” requirement or clearly correspond to patio chair cushions.

**Completeness:**  
Partial: The response does not consistently satisfy the colour constraint, and the structure is slightly inconsistent. One of the recommendations is presented outside of the main list, which makes it less clear that three products are being provided.

**Fluency:**  
Partial: The response is readable and generally understandable, but the structure is somewhat awkward. The model mixes product recommendations with commentary about limitations, which disrupts the flow.

**Other Observations:**  
This response reflects the model struggling to balance competing prompt constraints — specifically, providing at least three recommendations while also prioritizing well-supported and high-quality products. As a result, the output becomes slightly inconsistent in both structure and relevance.

---

### Query 3: something to make a garden look more colorful and lively

**Hybrid Output:**  

For a garden that looks more colorful and lively, I recommend the following products:

1. **Evergreen Garden Gazing Ball in Purple Iridescent Design Everyday Decor for Homes Gardens and Yards** ($59.94) - This beautiful gazing ball will add a pop of color and unique details to your garden. Its iridescent design will change with the seasons, making it a great addition to your outdoor decor.
2. **Life In Color Black Gnome Garden/Planter Decor (Rainbow)** - This little gnome will bring a burst of color and whimsy to your garden. Its vibrant colors will add a playful touch, and its durable resin material ensures it will withstand the elements.
3. **BP&Q Seasonal Garden Flag Set of 12 - Small Garden Flags 12x18 double sided for Outside - Outdoor Yard Flags Holiday Decorations for All Seasons with Anti-Wind Clip and Stopper** (Price: N/A) - This set of garden flags will add a splash of color to your garden with its cute designs for all seasons. You can change the flags with the seasons, ensuring your garden always looks lively and fresh.

If you're looking for something more unique, you might also consider the **HILUCK Welcome Snowman Home Garden Flag**, which will add a fun and festive touch to your garden during the winter months.

**Accuracy:**  
Yes: The recommendations are appropriate and aligned with the goal of adding color and vibrancy to a garden.

**Completeness:**  
Yes: The response fully addresses the query by offering a variety of decorative options that achieve the desired effect.

**Fluency:**  
Yes: The response is well-written and easy to follow.

**Other Observations:**
This response is particularly excellent - the only issue I see is that it included the price even though it is N/A. Otherwise it's an excellent example of RAG providing a strong improvement from the basic retrieval because it knows that the winter decor might not be relevant at the moment but shares it as an additional option with the context of "for winter months" that is super great. 

---

### Query 4: something to keep the deer out of my garden without hurting them

**Hybrid Output:**

Based on the provided reviews and metadata, here are some product recommendations to help keep deer out of your garden without hurting them:

1. **Deer Repel Deer Repellent Plants Pouches** ($34.97) - These pouches contain an all-natural formula that can deter deer and rabbits from eating plants, trees, and veggies. While they may not work on urban deer, they could be effective for country deer.
2. **Easy Gardener 6050AZ Netting and Fencing** ($24.9) - This durable netting can protect your garden from deer and other animals without harming them. It's easy to set up and can be reused.
3. **JFDWOPHT Reflective Pinwheels** ($9.99) - These pinwheels use reflective material to scare deer away without scaring or hurting them. They're a great, affordable option to try before investing in more substantial deterrents.

For a more proactive solution, you might consider the **Careland Solar Powered Ultrasonic Animal Repellent** ($45.99), which uses motion-activated lights and ultrasonic waves to scare deer away. However, its effectiveness may depend on the specific type of deer you're dealing with.

**Accuracy:**  
Yes: The recommendations are accurate and consistent with the requirement for humane deer deterrence.

**Completeness:**  
Yes: The response fully addresses the query by providing multiple solution types (repellent, barrier, deterrent devices).

**Fluency:**  
Yes: The response is clear, structured, and easy to understand.

**Other Observations:**
In the previous milestone this response resulted in a lot of extremely similar results. Now the LLM is effectively selecting without duplication as prompted - thats a strong improvement. It's also providing reasonable notes about limitations and considerations to keep in mind for each product.

---

### Query 5: what is a good plant to keep outside that won’t die if I forget to water it and doesn’t need much sunlight

**Hybrid Output:**  

Based on the provided context, I'd recommend the following plants for your outdoor space that are relatively low-maintenance and can tolerate some forgetfulness when it comes to watering:

1. **Costa Farms Yucca Cane** ($nan): This plant is known for its ability to thrive in low-light conditions and can survive with infrequent watering. It's a great option for a patio area that receives partial sunlight.
2. **9GreenBox -Tineke Rubber Tree Plant** ($7.99): Although the reviewer had a bad experience, the Rubber Tree Plant is known for its hardiness and ability to remove chemical toxins from the air. It can tolerate some neglect and requires minimal sunlight.

As for plants that don't require much sunlight, the 9GreenBox Rubber Tree Plant is a good option. However, if you're looking for a plant that can also tolerate some forgetfulness when it comes to watering, the Costa Farms Yucca Cane might be a better fit.

Keep in mind that all plants require some level of care, and it's always a good idea to check the soil moisture and adjust your watering schedule accordingly.

**Accuracy:**  
Partial: The recommendations are only partially accurate. While the plants are somewhat hardy, they do not both fully satisfy all constraints (low sunlight and low maintenance).

**Completeness:**  
No: The response is incomplete, as it does not fully address the multi-constraint nature of the query and it does not provide three responses as prompted.

**Fluency:**  
Yes: The response is clear and readable.

**Other Observations**:
This query failed with both semantic and BM25 retrieval in the previous milestone, so it's worth considering that the LLM was not given very helpful options from those search retrieval systems - with that in mind the performance isn't truly terrible as it was able to filter out the really unhelpful options that were retrieved to find two reasonable options. However, as noted they don't obviously meet all the requirements and there are only two provided. In this case, while it doesn't include 3 I think once again it is respecting the prompt direction to not share items that aren't useful.

---

## Key Observations

The hybrid RAG pipeline performs well for both structured and moderately open-ended queries. For keyword-based queries such as product specifications, the system consistently retrieves accurate and relevant results that fully satisfy the user’s request. For more abstract queries, the hybrid approach demonstrates strong performance by returning a diverse set of recommendations that capture different interpretations of the query. 

However, performance declines for more complex queries involving multiple constraints. In these cases, the system often produces only partially relevant results, highlighting the difficulty of satisfying nuanced user intent using retrieval-based context alone. This is likely due to the actual low quality of the retrieved documents shared for context to the LLM, in those cases while it failed to provide 3 products as directed it actually seems to be reasonably filtering out items that are not appropriate for the prompt. 

---

## Limitations

One limitation of the hybrid RAG workflow is its sensitivity to prompt size. Because it combines results from both semantic and BM25 retrieval, the resulting context can become large and exceed the token limits of the LLM. To address this, we truncated longer text fields and reduced the number of retrieved documents, which may limit the amount of information available to the model.

A second limitation is the system’s reliance on the available dataset and the success of the retrieval systems that feed it context. When relevant products are not well represented in the retrieved context, the model produces approximate or partially relevant recommendations. This is particularly evident in complex queries requiring multiple constraints.

---

## Suggestions for Improvement

One potential improvement would be to incorporate a re-ranking step after retrieval to better prioritize the most relevant documents before passing them to the language model. This could improve both accuracy and completeness, especially in hybrid retrieval settings.

Another improvement would be to make context construction more adaptive. For example, dynamically selecting which fields to include or adjusting the level of truncation based on the query type could help balance information richness with token constraints.

---

## Summary

Overall, the hybrid RAG pipeline demonstrates strong performance across a range of query types, particularly by improving result diversity compared to semantic-only retrieval. While it remains limited by dataset coverage and token constraints, it provides a practical and effective approach for generating grounded product recommendations.