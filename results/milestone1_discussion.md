# Milestone 1 Retrieval Evaluation

This document outlines 10 queries that were tested on both the BM25 and Semantic retrieval methods. The code to run all 10 queries can be found in the notebooks/milestone1_evaluation.ipynb file.

## Query Set

**Easy (keyword-based)**

1. blue patio chair cushion  
2. left handed stainless steel garden trowel
3. 6-inch ceramic plant pot with drainage hole

**Medium (semantic-based)**

4. something to make a garden look more colorful and lively  
5. tool for cutting back a large tree
6. something non-toxic to protect my flowers from deer
7. cool gift for someone who loves gardening

**Complex (may require RAG)**

8. plant suitable for moving outdoors to a patio in spring  
9. best outdoor decoration that won’t fade in the sun  
10. good plant to keep outside that won’t die if I forget to water it and doesn’t need much sunlight

---

## Detailed Comparison (5 Selected Queries)

### Query 1: "blue patio chair cushion" 

**BM25 Top 5**
1. Rocking Chair Cushion Pad, 47IN Indoor Outdoor Navy Blue Rocking Chair Cushions Set Soft Thickened Patio Chaise Lounger Cushion Overstuffed Patio Chair Cushion  
2. Rocking Chair Cushion Pad, 2 Pieces Indoor/Outdoor Rocking Chair Cushions Set Indoor/Outdoor Soft Thickened Patio Chaise Lounger Cushion Overstuffed Patio Chair Cushion (Navy Blue)  
3. REDCAMP Chaise Lounge Chair Cushion for Patio Furniture, Thicker Soft Comfortable Zero Gravity Chair Pad for Outdoor Indoor Home Office, Blue 65"x21"  
4. Comfort Classics Inc. 22W x 44L x 5H Hinge at 24" Sunbrella Outdoor CHANNELED Chair Cushion in Air Blue  
5. Pillow Perfect Outdoor/Indoor Basalto Navy Square Corner Chair Cushion, 1 Count (Pack of 1), Blue  

**Semantic Top 5**
1. Pillow Perfect Stripe Indoor/Outdoor 1 Piece Split Back Round Corner Chair Seat Cushion with Ties, Deep Seat, Weather, and Fade Resistant, 40.5" x 21", Red/White Midland, 1 Count  
2. Rocking Chair Cushion Pad, 47IN Indoor Outdoor Navy Blue Rocking Chair Cushions Set Soft Thickened Patio Chaise Lounger Cushion Overstuffed Patio Chair Cushion  
3. LVTXIII Outdoor Seat Cushions Patio Chair Pads with Ties, All Weather Chair Cushions for Home Office and Patio Garden Furniture Decoration D16”xW17”, Stripe Beige, Set of 2  
4. Indoor/Outdoor Bench Cushion Cotton Garden Furniture Loveseat Cushion, 51.2"x19.7" Patio Wicker Seat Cushions for Lounger Garden Furniture Patio Lounger Bench (Navy Blue)  
5. Amazon Basics Deep Seat Patio Seat and Back Cushion Set - Black Floral  

**Comparison**  
In this query, the BM25 method clearly outperforms the semantic method due to the importance of the specific attribute "blue." BM25 retrieves results that consistently match this keyword, with all top results being blue cushions. In contrast, semantic search fails to preserve the color constraint, returning items that do not match the user’s requirement, including pillows that are red, beige, or patterned. This highlights that BM25 is more effective when exact attributes are critical, whereas semantic search may sacrifice precision for broader similarity.

---

### Query 2: "6-inch ceramic plant pot with drainage hole"

**BM25 Top 5**
1. Small Plant Pots Set, 4.6 Inch & 6 Inch Ceramic Planter Pot for Plants with Drainage Hole and Saucer, Speckled Grey, 94-G-S-3  
2. D'vine Dev 6 Inch Ceramic Planter Pot with Drainage Hole and Saucer, Indoor Cylinder Round Planter Pot, Black/Speckled Tan, 94-O-S-7  
3. D'vine Dev 6 Inch Ceramic Planter Pot with Drainage Hole and Saucer, Indoor Cylinder Round Planter Pot, Black/Speckled Tan, 94-O-S-7  
4. ADN Ceramic Planter, 6 Inch Plant Pots for Plants Modern Decor Indoor Flower Pot with Saucer and Drainage Hole, White  
5. Giftacity Toilet Planter Pot, Novelty Mini Planter with Drainage Hole and Ceramic Tray, Funny Cute Succulent Plant Container for Home Bathroom Décor  

**Semantic Top 5**
1. ADN Ceramic Planter, 6 Inch Plant Pots for Plants Modern Decor Indoor Flower Pot with Saucer and Drainage Hole, White  
2. DEARDESIRES 10 inch Ceramic Plant Pot, Indoor Outdoor White Flower Pot, Large Planter fits 9 or 8 inch Nursery Pot, Ceramic Planter with 2 Drainage Holes and Plugs, Gloss White, Wavy Texture  
3. Ufrount 3 Inches Ceramic Planter Pot with Drainage Holes, Succulent Planter Pots Planting Pot Flower Pots for Mini Plant Perfect for Garden, Kitchen, Windowsill - Set of 6  
4. Gohearin Ceramic Plant Flowerpot, Hand-Painted Flowerpot Set, 6.6+5.4-inch Indoor Flowerpot with Drain Hole, Used for Indoor Plants, Cactus, Succulent Plants and Other Outdoor Garden, 2-Piece Set  
5. YFFSRJDJ Ceramic Flower Plants Pots Planter with Drainage Hole, 6.0 inch+5.0 Inch. Indoor-Outdoor Large Round Succulent Orchid Pot Set (Blue)  

**Comparison**  
In this query, BM25 again performs better, but for a slightly different reason: the query contains multiple structured constraints (size, material, and the presence of a drainage hole). BM25 is able to match these terms directly and returns results that satisfy all key requirements. In contrast, the semantic search retrieves generally similar items but does not consistently enforce these constraints, particularly the exact size. This suggests that BM25 is particularly effective when queries include multiple specific product specifications that must all be satisfied simultaneously.

---

### Query 3: "something to make a garden look more colorful and lively"  

**BM25 Top 5**
1. BP&Q Seasonal Garden Flag Set of 12 - Small Garden Flags 12x18 double sided for Outside - Outdoor Yard Flags Holiday Decorations for All Seasons with Anti-Wind Clip and Stopper  
2. BP&Q Seasonal Garden Flag Set of 12 - Small Garden Flags 12x18 double sided for Outside - Outdoor Yard Flags Holiday Decorations for All Seasons with Anti-Wind Clip and Stopper  
3. HILUCK Welcome Snowman Home Garden Flag, Let It Snow Cute Little Snowman Decoration, Burlap Vertical Double Sided for Winter Merry Christmas Yard Celebration Banner in Lawn Outdoor 12 x 18 inch  
4. Seasonal Garden Flags Set of 12 Double Sided Burlap 12.5 x 18 Inch House Flags ,Small Garden Flags for Outside,Independence Day Flag,Fall Garden Flags,Halloween Garden Flags,Thanksgiving Garden Flag for Outdoor Decorations Flags  
5. Seasonal Garden Flags Set of 12 Double Sided Burlap 12.5 x 18 Inch House Flags ,Small Garden Flags for Outside,Independence Day Flag,Fall Garden Flags,Halloween Garden Flags,Thanksgiving Garden Flag for Outdoor Decorations Flags  

**Semantic Top 5**
1. Evergreen Garden Gazing Ball in Purple Iridescent Design Everyday Decor for Homes Gardens and Yards  
2. Life In Color Black Gnome Garden/Planter Decor (Rainbow)  
3. Exhart Garden Solar Lights, Set of 2 Decorative Bubble Garden Stakes, Color Changing LED, Outdoor Garden and Yard Decoration, 2 x 30 Inch, Square  
4. Evergreen Garden Gazing Ball in Purple Iridescent Design Everyday Decor for Homes Gardens and Yards  
5. Little Garden Gnome 300 Pcs Glow in The Dark Pebbles for Walkways and Decor | Decorative Stones for Gardens, Yards, Lawns, Driveways, Plants, Aquarium | Electric Blue  

**Comparison**
In this query, the semantic search clearly outperforms BM25 because the query is based on an abstract, descriptive goal rather than specific product keywords. BM25 fails to capture the intended aesthetic goal, instead returning generic and repetitive items (e.g., seasonal flags) that do not strongly reflect the intent of making a garden more “colorful and lively.” In contrast, the semantic search captures this intent and retrieves visually relevant items such as colorful ornaments, lights, and decorative features. This demonstrates that semantic search is better suited for queries that require interpreting overall meaning and intent, rather than relying on exact keyword matches.

---

### Query 4: "cool gift for someone who loves gardening"

**BM25 Top 5**
1. Gorilla Grip Original Spa Bath Pillow Features Powerful Gripping Technology, Comfortable, Soft, Large, 19.5x15, Luxury 3-Panel Design for Shoulder, Neck Support, Fits Any Size Tub, Jacuzzi, Spas  
2. UP THE MOMENT Plant Lady Hat, Plant Lady Gift, Succulent Plants Gift, Garden Gifts for Women, Plant Lover Gifts, Plant Gift, Gifts for Gardeners Women, Plant Gifts for Women Olive, Khaki  
3. Joan Baker Designs TP1027 Tile Plaque, Dragonfly and Lilies/Love Joy Peace, 6 by 7-Inch  
4. Colsen Custom Tabletop Rubbing Alcohol Fireplace Indoor Outdoor Fire Pit Portable Fire Concrete Bowl Pot Fireplace (Rectangular) (Classic)  
5. Mr. Bar-B-Q Deluxe BBQ Tool Set | All in One BBQ Tool Set | Premium Hard-Shell Case | Contains 18 Stainless Steel BBQ Grilling Tools | BBQ Tools Set for Men  

**Semantic Top 5**
1. Funny Combat Garden Gnome Riding a Flamingo – Cute Home Decor Statue, for Lawn, Yard, or Office - Indoor Outdoor 11 Inch Tall  
2. Garden Set Birthday Gifts for Mom. Wearable Garden Tool Set with Knee Pad & Garden Tools  
3. Mood Lab Garden Gnome - Zen Gnome Statue - 9.25 Inch Tall Lawn Gnome Figurine  
4. Vegtrug Limited Patio Garden  
5. TERESA'S COLLECTIONS Ladybug Garden Decor with Solar Lights  

**Comparison**  
In this query, semantic search clearly outperforms BM25 because the query expresses a specific user context and purpose (finding a gift for someone who loves to garden), rather than a direct description of a product. BM25 fails to capture the gardening context, returning loosely related or irrelevant items that do not reflect the gardening theme. In contrast, the semantic search interprets the intended use case and retrieves items that are appropriate as gifts within the gardening domain, such as decorative items and tool sets. This demonstrates that semantic search is particularly effective at mapping user intent to relevant product categories, even when the query does not explicitly specify them.

---

### Query 5: "good plant to keep outside that won’t die if I forget to water it and doesn’t need much sunlight"

**BM25 Top 5**
1. SONKIR Soil pH Tester, 3-in-1 Soil Moisture/Light/pH Tester Gardening Tool Kits  
2. TomCare Garden Hose Holder Detachable Metal Water Hose Holder  
3. 3PCS White Pink Mini Size Lithops Rare Lithops meyeri Small Succulent Plants  
4. Brown -Yellow Dragon Pattern Amber Mini Size Lithops Rare Lithops meyeri Small Succulent Plants  
5. Orchid Pots with Holes - 8 Pack 7 inch Orchid Pots for Repotting  

**Semantic Top 5**
1. Luffy Coco Philippines Java Fern: Live Aquatic Plant with 10+ Leaves  
2. Luffy Coco Philippines Java Fern: Live Aquatic Plant with 10+ Leaves  
3. Bloomify Live Sundew Plant Terrarium – Enclosed Ecosystem with Zero Maintenance  
4. Luffy Coco Philippines Java Fern: Live Aquatic Plant with 10+ Leaves  
5. Luffy Coco Philippines Java Fern: Live Aquatic Plant with 10+ Leaves  

**Comparison**
This query demonstrates a level of complexity where both the BM25 and semantic retrieval methods fail. The BM25 method returns largely irrelevant items (e.g., tools and accessories) that do not address the core intent of selecting an appropriate plant, indicating that it cannot handle multi-part constraints beyond simple keyword matches. The semantic search performs slightly better in retrieving only plants, but fails to satisfy the full set of conditions, instead overemphasizing one aspect of the query (e.g., low maintenance with respect to watering) and returns inappropriate results such as aquatic plants. Overall, neither method successfully captures the combined requirements of a outdoor plant with low watering needs and low sunlight tolerance. This highlights a key limitation of both approaches when dealing with complex, multi-constraint queries, suggesting that more advanced methods such as retrieval-augmented generation (RAG) may be needed. 

---

### Summary Insights

From this analysis, we observe clear differences in the strengths and weaknesses of BM25 and semantic search. BM25 performs best for queries that require exact matching of specific attributes, such as colour, size, or other structured product specifications. In these cases, semantic search often fails to strictly preserve these constraints, instead returning items that are broadly relevant but do not fully match the query requirements. In contrast, semantic search performs better for queries that involve abstract or descriptive language, where the user’s intent must be interpreted rather than directly matched. For example, when queries describe a goal or use case (e.g., making a garden more “colorful and lively” or finding a suitable gift), semantic search is able to retrieve more relevant results. In these situations, BM25 fails because it prioritizes exact keyword overlap, which is insufficient for capturing the intended meaning behind the query. Finally, both BM25 and semantic search struggle with more complex queries that involve multiple constraints or require real-world knowledge. When queries combine several conditions (e.g., low maintenance, low sunlight, outdoor suitability), neither method is able to fully satisfy all requirements. This suggests that more advanced approaches, such as retrieval-augmented generation (RAG), may be needed to handle these types of queries effectively.