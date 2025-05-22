"""
Search functionality for the AI Product Expert Bot.
This script provides functions to search for products using text or image queries.
"""

import os
import logging
import argparse
from typing import Dict, List, Optional, Any, Union
import numpy as np
import json
import hashlib
import functools
import time
from fastembed import SparseTextEmbedding # For MiniCOIL
from qdrant_client.http import models

# Import our modules
import config
import vector_store
from gcp_clients import get_multimodal_embeddings
from gemini_client import generate_text as generate_gemini_text
from text_embedding_client import get_text_embedding as get_dedicated_text_embedding
from data_ingestion import encode_categorical_n_hot, encode_price_normalized # Add these back

# Vocabularies and stats will be loaded or built
BRAND_VOCAB: List[str] = []
PRODUCT_TYPE_VOCAB: List[str] = []
AVAILABILITY_VOCAB: List[str] = ["in_stock", "out_of_stock", "preorder", "backorder"] # Defined here, could also be in data_ingestion
PRICE_STATS = {"min": 0.0, "max": 1.0, "count": 0, "sum": 0.0} # Default
PRODUCT_TYPE_VOCAB_EMBEDDINGS: Dict[str, List[float]] = {}

VOCABS_LOADED_SUCCESSFULLY = False

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Sparse Embedding Model for search queries
try:
    sparse_embedding_model_search = SparseTextEmbedding(model_name="Qdrant/minicoil-v1")
    logger.info("SparseTextEmbedding model (Qdrant/minicoil-v1) initialized successfully for search.")
except Exception as e:
    logger.error(f"Failed to initialize SparseTextEmbedding model for search: {e}", exc_info=True)
    sparse_embedding_model_search = None

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Computes cosine similarity between two vectors.

    Args:
        vec1: The first vector (list of floats).
        vec2: The second vector (list of floats).

    Returns:
        The cosine similarity score (float) between the two vectors.
        Returns 0.0 if either vector is empty or if a norm is zero.
    """
    if not vec1 or not vec2: return 0.0
    vec1_arr = np.array(vec1)
    vec2_arr = np.array(vec2)
    dot_product = np.dot(vec1_arr, vec2_arr)
    norm_vec1 = np.linalg.norm(vec1_arr)
    norm_vec2 = np.linalg.norm(vec2_arr)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

def ensure_vocabs_for_search():
    """
    Ensures that vocabularies (BRAND_VOCAB, PRODUCT_TYPE_VOCAB), price statistics (PRICE_STATS),
    and product type embeddings (PRODUCT_TYPE_VOCAB_EMBEDDINGS) are loaded.

    It first tries to load these from a persisted JSON file (`vocab_embeddings.json`).
    If the file doesn't exist, is incomplete, or an error occurs during loading,
    it attempts to rebuild these artifacts by calling functions from the `data_ingestion` module.
    This function is critical for search operations that rely on these vocabularies and embeddings
    for query vector construction and attribute matching.

    Sets the global `VOCABS_LOADED_SUCCESSFULLY` flag.
    """
    global BRAND_VOCAB, PRODUCT_TYPE_VOCAB, PRICE_STATS, PRODUCT_TYPE_VOCAB_EMBEDDINGS, VOCABS_LOADED_SUCCESSFULLY
    
    if VOCABS_LOADED_SUCCESSFULLY:
        return

    # Try to load from persisted file first (created by data_ingestion.py)
    from data_ingestion import VOCAB_EMBEDDINGS_FILE # Get filename
    if os.path.exists(VOCAB_EMBEDDINGS_FILE):
        try:
            with open(VOCAB_EMBEDDINGS_FILE, 'r') as f:
                data = json.load(f)
            BRAND_VOCAB = data.get("BRAND_VOCAB", [])
            PRODUCT_TYPE_VOCAB = data.get("PRODUCT_TYPE_VOCAB", [])
            PRICE_STATS = data.get("PRICE_STATS", PRICE_STATS)
            PRODUCT_TYPE_VOCAB_EMBEDDINGS = data.get("PRODUCT_TYPE_VOCAB_EMBEDDINGS", {})
            
            if BRAND_VOCAB and PRODUCT_TYPE_VOCAB and PRICE_STATS["count"] > 0 and PRODUCT_TYPE_VOCAB_EMBEDDINGS:
                logger.info(f"Successfully loaded vocabs and embeddings from {VOCAB_EMBEDDINGS_FILE} in search.py")
                logger.info(f"Brands: {len(BRAND_VOCAB)}, Types: {len(PRODUCT_TYPE_VOCAB)}, Type Embeddings: {len(PRODUCT_TYPE_VOCAB_EMBEDDINGS)}")
                VOCABS_LOADED_SUCCESSFULLY = True
                return
            else:
                logger.warning(f"Persisted file {VOCAB_EMBEDDINGS_FILE} was incomplete. Will try to rebuild.")
        except Exception as e:
            logger.error(f"Error loading {VOCAB_EMBEDDINGS_FILE} in search.py: {e}. Will try to rebuild.")

    # If loading failed or file doesn't exist, try to trigger build via data_ingestion
    logger.warning("Vocabularies or embeddings not loaded. Attempting to build them by calling data_ingestion.")
    try:
        import data_ingestion as di
        # This will run the main() of data_ingestion if it's not guarded by if __name__ == "__main__"
        # Or, more safely, call its build function directly if it's refactored to allow that.
        # For now, assume data_ingestion.py needs to be run once to create the file.
        # A better way is for data_ingestion to have a callable function to build/load.
        # Let's call its build_vocabularies_and_stats directly.
        feed_url = "https://files.channable.com/gOBn8ftRlhqTvgdOkoBnIw==.xml" # Needs to be accessible
        xml_content = di.download_xml_feed(feed_url)
        if xml_content:
            products = di.parse_xml_feed(xml_content)
            if products:
                di.build_vocabularies_and_stats(products, force_rebuild=True) # Force rebuild to create the file
                # Now try loading again
                if os.path.exists(VOCAB_EMBEDDINGS_FILE):
                    with open(VOCAB_EMBEDDINGS_FILE, 'r') as f: data = json.load(f)
                    BRAND_VOCAB = data.get("BRAND_VOCAB", [])
                    PRODUCT_TYPE_VOCAB = data.get("PRODUCT_TYPE_VOCAB", [])
                    PRICE_STATS = data.get("PRICE_STATS", PRICE_STATS)
                    PRODUCT_TYPE_VOCAB_EMBEDDINGS = data.get("PRODUCT_TYPE_VOCAB_EMBEDDINGS", {})
                    if BRAND_VOCAB and PRODUCT_TYPE_VOCAB and PRICE_STATS["count"] > 0 and PRODUCT_TYPE_VOCAB_EMBEDDINGS:
                         logger.info("Successfully built and loaded vocabs/embeddings after fallback.")
                         VOCABS_LOADED_SUCCESSFULLY = True
                         return
        logger.error("Fallback: Failed to build/load necessary vocabularies/stats in search.py.")
    except Exception as e:
        logger.error(f"Exception during fallback vocab build in search.py: {e}")

    # If still not loaded, set flag
    VOCABS_LOADED_SUCCESSFULLY = False


def parse_query_with_llm(natural_language_query: str) -> Dict[str, Any]:
    """
    Parses a natural language query using a Large Language Model (Gemini) to extract
    structured information for search.

    The LLM is prompted to identify:
    - Text segments for description and image modality matching.
    - Keywords for sparse search.
    - Query intent (e.g., search, recommendation).
    - Product attributes (brand, product type, price, availability).
    - Weights for different components of the dense query vector.

    Args:
        natural_language_query: The user's query string.

    Returns:
        A dictionary containing the parsed query components.
        Includes keys like "search_text_for_description", "attributes", "weights".
        Returns a fallback dictionary if LLM parsing fails or API key is not set.
    """
    ensure_vocabs_for_search()
    # ... (rest of the function remains largely the same, but the prompt can be simpler
    #      as we don't need to provide as many examples if using embedding similarity for product_type)

    brand_examples_str = ", ".join(BRAND_VOCAB[:5]) + "..." if len(BRAND_VOCAB) > 5 else ", ".join(BRAND_VOCAB)
    # type_examples_str = ", ".join(PRODUCT_TYPE_VOCAB[:5]) + "..." if len(PRODUCT_TYPE_VOCAB) > 5 else ", ".join(PRODUCT_TYPE_VOCAB)
    # No longer need to give type examples if we use embedding similarity

    prompt = f"""
    You are an intelligent query understanding system for an e-commerce product search with multimodal capabilities.
    Parse the user's query to extract structured information for searching. The goal is to create a query vector with separate segments for text description, image-related aspects, brand, product type, availability, and price.

    User Query: "{natural_language_query}"

    Output a JSON object with the following fields:
    - "search_text_for_description": (string) Text primarily for matching product titles and descriptions (for dense vector).
    - "search_text_for_image_modality": (string, optional) Text describing visual attributes for text-to-image search (for dense vector's image segment).
    - "keyword_search_text": (string, optional) Concise keywords from the query for precise term matching (for sparse vector). If not distinct, can be same as search_text_for_description.
    - "query_intent": (string, optional) Can be "search", "recommendation", "comparison". Infer from query.
    - "attributes": (object, optional)
        - "brand": (string, optional) e.g., {brand_examples_str}
        - "product_type": (string, optional) General primary product type/category (e.g., "Grill", "Gift Basket").
        - "availability": (string, optional) "in_stock", "out_of_stock", etc.
        - "price_min": (float, optional)
        - "price_max": (float, optional) If user mentions a "budget of XX", this is price_max.
        - "price_target": (float, optional)
        - "price_qualitative": (string, optional) "cheap", "moderate", "expensive".
    - "weights": (object) Importance of components (0.0-1.0) for the DENSE vector. ALWAYS include all dense weight keys below, using a low default (e.g., 0.1) if not central. (Sparse vector search has its own scoring, not directly weighted here by LLM).
        - "description_text_weight": (float)
        - "image_modality_text_weight": (float)
        - "query_image_weight": (float)
        - "brand_weight": (float)
        - "product_type_weight": (float)
        - "availability_weight": (float)
        - "price_weight": (float)

    HANDLING COMPLEX QUERIES:
    - For long, detailed queries, extract the most important search terms and attributes
    - For queries with multiple constraints, identify all constraints and assign appropriate weights
    - For queries with contradictory information, prioritize the most specific or recent information

    HANDLING IMAGE DESCRIPTIONS:
    - For queries about visual appearance, put detailed visual attributes in "search_text_for_image_modality"
    - Include colors, materials, patterns, shapes, and styles in image modality text
    - Assign higher "image_modality_text_weight" for queries focused on appearance

    HANDLING IMAGE-ONLY QUERIES:
    - When no text query is provided (empty string), this indicates an image-only search
    - Set "query_image_weight" to 0.9 for image-only searches to prioritize visual similarity
    - Use low weights (0.1) for all other components since no text context is available
    - Leave text fields empty or minimal

    Example for image-only search (empty query with uploaded image):
    {{
        "search_text_for_description": "",
        "search_text_for_image_modality": "",
        "keyword_search_text": "",
        "query_intent": "search",
        "attributes": {{}},
        "weights": {{
            "description_text_weight": 0.1,
            "image_modality_text_weight": 0.1,
            "query_image_weight": 0.9,
            "brand_weight": 0.1,
            "product_type_weight": 0.1,
            "availability_weight": 0.1,
            "price_weight": 0.1
        }}
    }}

    Example for "cheap OFYR grill with a rustic look":
    {{
        "search_text_for_description": "OFYR grill outdoor cooking rustic",
        "search_text_for_image_modality": "rustic OFYR grill corten steel look",
        "keyword_search_text": "OFYR grill rustic steel",
        "query_intent": "search",
        "attributes": {{ "brand": "OFYR", "product_type": "Outdoor Grill", "price_qualitative": "cheap" }},
        "weights": {{
            "description_text_weight": 0.7,
            "image_modality_text_weight": 0.6,
            "query_image_weight": 0.1,
            "brand_weight": 0.9,
            "product_type_weight": 0.8,
            "price_weight": 0.7,
            "availability_weight": 0.1
        }}
    }}

    Example for "birthday gift for mom, budget 500 DKK, something nice":
    {{
        "search_text_for_description": "nice birthday gift for mom",
        "search_text_for_image_modality": "elegant gift",
        "keyword_search_text": "birthday gift mom nice",
        "query_intent": "recommendation",
        "attributes": {{ "price_max": 500.0 }},
        "weights": {{
            "description_text_weight": 0.8,
            "image_modality_text_weight": 0.5,
            "query_image_weight": 0.1,
            "brand_weight": 0.2,
            "product_type_weight": 0.5,
            "price_weight": 0.9,
            "availability_weight": 0.3
        }}
    }}

    Example for a complex query "I'm looking for a high-quality coffee machine for my small office kitchen, preferably from Miele or Siemens, that can make both espresso and regular coffee, has a built-in grinder, and costs between 3000-5000 DKK. It should match our modern stainless steel appliances.":
    {{
        "search_text_for_description": "high-quality coffee machine office espresso regular coffee built-in grinder",
        "search_text_for_image_modality": "modern stainless steel coffee machine",
        "keyword_search_text": "coffee machine espresso grinder Miele Siemens",
        "query_intent": "search",
        "attributes": {{
            "brand": "Miele Siemens",
            "product_type": "Coffee Machine",
            "price_min": 3000.0,
            "price_max": 5000.0
        }},
        "weights": {{
            "description_text_weight": 0.9,
            "image_modality_text_weight": 0.7,
            "query_image_weight": 0.1,
            "brand_weight": 0.8,
            "product_type_weight": 0.9,
            "price_weight": 0.8,
            "availability_weight": 0.5
        }}
    }}
    
    Parse the user query. Output valid JSON. Ensure all DENSE weight keys are present in the "weights" object.
    """
    if not config.GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set. LLM parsing fallback.")
        return {"search_text": natural_language_query, "attributes": {}, "weights": {}}

    llm_response_str = generate_gemini_text(prompt)
    logger.info(f"LLM response for query parsing: {llm_response_str}")
    try:
        # More robust JSON extraction
        json_start = llm_response_str.find('{')
        if json_start == -1:
            raise ValueError("No JSON object found in response")
        
        # Find matching closing brace
        brace_count = 0
        json_end = -1
        for i in range(json_start, len(llm_response_str)):
            if llm_response_str[i] == '{':
                brace_count += 1
            elif llm_response_str[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break
        
        if json_end == -1:
            raise ValueError("No matching closing brace found")
        
        json_str = llm_response_str[json_start:json_end]
        parsed_json = json.loads(json_str)
        
        # Validate response structure
        if not isinstance(parsed_json, dict):
            raise ValueError("Parsed JSON is not a dictionary")
        
        # Ensure primary dense text field has a fallback
        if "search_text_for_description" not in parsed_json or not parsed_json["search_text_for_description"]:
            parsed_json["search_text_for_description"] = natural_language_query
            logger.info(f"LLM did not provide 'search_text_for_description', using full query: '{natural_language_query}'")
        
        # Ensure required fields exist
        if "attributes" not in parsed_json:
            parsed_json["attributes"] = {}
        if "weights" not in parsed_json:
            parsed_json["weights"] = {}
        
        return parsed_json
    except Exception as e:
        logger.error(f"Failed to parse LLM response: {e}. Using fallback.")
        return {
            "search_text_for_description": natural_language_query,
            "attributes": {},
            "weights": {}
        }


def find_best_matching_product_type(llm_suggested_type: Optional[str], similarity_threshold: float = 0.6) -> Optional[str]:
    """
    Finds the best matching product type from the project's vocabulary (PRODUCT_TYPE_VOCAB)
    for a product type suggested by the LLM.

    This function embeds the LLM's suggested product type and compares its embedding
    against pre-computed embeddings for all product types in the vocabulary.
    The comparison uses cosine similarity.

    Args:
        llm_suggested_type: The product type string suggested by the LLM.
        similarity_threshold: The minimum cosine similarity score required for a match.

    Returns:
        The best matching product type string from the vocabulary if a match above the
        threshold is found; otherwise, None.
    """
    if not llm_suggested_type or not PRODUCT_TYPE_VOCAB_EMBEDDINGS or not VOCABS_LOADED_SUCCESSFULLY:
        return None

    query_type_embedding = get_dedicated_text_embedding(llm_suggested_type)
    if not query_type_embedding:
        logger.warning(f"Could not embed LLM suggested type: {llm_suggested_type}")
        return None

    best_match_term = None
    highest_similarity = -1.0

    for vocab_term, vocab_embedding in PRODUCT_TYPE_VOCAB_EMBEDDINGS.items():
        sim = cosine_similarity(query_type_embedding, vocab_embedding)
        if sim > highest_similarity:
            highest_similarity = sim
            best_match_term = vocab_term
            
    if best_match_term and highest_similarity >= similarity_threshold:
        logger.info(f"Product type match: LLM='{llm_suggested_type}' -> Vocab='{best_match_term}' (Similarity: {highest_similarity:.4f})")
        return best_match_term
    else:
        logger.info(f"No sufficiently similar product type found for '{llm_suggested_type}'. Highest sim: {highest_similarity:.4f} with '{best_match_term}' (Threshold: {similarity_threshold})")
        return None


def construct_holistic_query_vector(
    parsed_query: Dict[str, Any],
    image_path_for_query: Optional[str] = None
) -> Optional[List[float]]:
    """
    Constructs a holistic dense query vector based on the LLM-parsed query and an optional image.

    The holistic vector is a concatenation of several weighted and normalized segments:
    1.  Text embedding (from `parsed_query["search_text_for_description"]`).
    2.  Image embedding (from `image_path_for_query` or `parsed_query["search_text_for_image_modality"]`).
    3.  Brand vector (n-hot encoded based on `parsed_query["attributes"]["brand"]`).
    4.  Product type vector (n-hot encoded based on matched `parsed_query["attributes"]["product_type"]`).
    5.  Availability vector (n-hot encoded based on `parsed_query["attributes"]["availability"]`).
    6.  Price vector (normalized scalar based on `parsed_query["attributes"]["price_*"]`).

    Each segment is individually L2 normalized before being weighted and concatenated.
    The weights are derived from `parsed_query["weights"]`.

    Args:
        parsed_query: The dictionary output from `parse_query_with_llm`.
        image_path_for_query: Optional path to an image file to include in the query.

    Returns:
        A list of floats representing the final concatenated holistic query vector,
        or None if vocabularies are not loaded or a dimension mismatch occurs.
    """
    ensure_vocabs_for_search()
    if not VOCABS_LOADED_SUCCESSFULLY:
        logger.error("Cannot construct query vector: Vocabs not loaded.")
        return None

    # Get parsed components from LLM output
    desc_search_text = parsed_query.get("search_text_for_description", "")
    image_modality_search_text = parsed_query.get("search_text_for_image_modality")
    attributes = parsed_query.get("attributes", {})
    llm_weights = parsed_query.get("weights", {})

    # Define weights, using LLM suggestions or very low defaults for unspecified structured attributes
    default_low_structured_weight = 0.01 # For structured attributes not explicitly weighted by LLM
    default_active_structured_weight = 0.5 # Default if LLM mentions attribute but not weight
    
    w_desc_text = llm_weights.get("description_text_weight", 0.6) # Default if LLM omits (aligning with prompt)
    w_img_mod_text = llm_weights.get(
        "image_modality_text_weight",
        0.4 if image_modality_search_text else 0.0 # Use 0.4 if text for image modality is present but no specific weight
    )
    w_query_img = llm_weights.get("query_image_weight", 0.1) # Default 0.1 (aligning with prompt)

    # For structured attributes, if LLM mentions them, use its weight or a default active one.
    # If not mentioned by LLM at all, use a very low weight or zero.
    w_brand = llm_weights.get("brand_weight", default_active_structured_weight if attributes.get("brand") else default_low_structured_weight)
    w_ptype = llm_weights.get("product_type_weight", default_active_structured_weight if attributes.get("product_type") else default_low_structured_weight)
    w_avail = llm_weights.get("availability_weight", default_active_structured_weight if attributes.get("availability") else default_low_structured_weight)
    price_attr_present = any(k in attributes for k in ["price_target", "price_min", "price_max", "price_qualitative"])
    w_price = llm_weights.get("price_weight", default_active_structured_weight if price_attr_present else default_low_structured_weight)

    # 1. Prepare Text Embedding Segment (for product descriptions)
    query_desc_text_embedding = None
    if desc_search_text:
        _, query_desc_text_embedding = get_multimodal_embeddings(contextual_text=desc_search_text)
    
    text_segment_vector = [0.0] * config.MULTIMODAL_EMBEDDING_DIMENSION
    if query_desc_text_embedding:
        norm_desc_text = np.linalg.norm(query_desc_text_embedding)
        if norm_desc_text > 0:
            text_segment_vector = [x / norm_desc_text for x in query_desc_text_embedding]
    text_segment_vector = [x * w_desc_text for x in text_segment_vector]
    logger.info(f"Applied description_text_weight: {w_desc_text}")

    # 2. Prepare Image Embedding Segment
    image_segment_vector = [0.0] * config.MULTIMODAL_EMBEDDING_DIMENSION
    actual_query_image_embedding = None
    if image_path_for_query and os.path.exists(image_path_for_query):
        actual_query_image_embedding, _ = get_multimodal_embeddings(image_path=image_path_for_query)
        if actual_query_image_embedding:
            norm_actual_img = np.linalg.norm(actual_query_image_embedding)
            if norm_actual_img > 0:
                current_image_segment_source = [x / norm_actual_img for x in actual_query_image_embedding]
            image_segment_vector = [x * w_query_img for x in current_image_segment_source]
            logger.info(f"Using actual query image for image segment, applied query_image_weight: {w_query_img}")
    
    if not actual_query_image_embedding and image_modality_search_text: # Fallback to text-for-image
        _, query_img_mod_text_embedding = get_multimodal_embeddings(contextual_text=image_modality_search_text)
        if query_img_mod_text_embedding:
            norm_img_text = np.linalg.norm(query_img_mod_text_embedding)
            if norm_img_text > 0:
                image_segment_vector = [x / norm_img_text for x in query_img_mod_text_embedding]
            image_segment_vector = [x * w_img_mod_text for x in image_segment_vector]
            logger.info(f"Using text-for-image-modality for image segment, applied image_modality_text_weight: {w_img_mod_text}")

    # 3. Encode Structured Attributes
    brand_query = attributes.get("brand")
    brand_vector = encode_categorical_n_hot(brand_query, BRAND_VOCAB)
    norm_brand = np.linalg.norm(brand_vector)
    if norm_brand > 0: brand_vector = [x / norm_brand for x in brand_vector]
    brand_vector = [x * w_brand for x in brand_vector]
    logger.info(f"Applied brand_weight: {w_brand}")

    llm_suggested_pt = attributes.get("product_type")
    matched_pt_vocab_term = find_best_matching_product_type(llm_suggested_pt)
    product_type_vector = encode_categorical_n_hot(matched_pt_vocab_term, PRODUCT_TYPE_VOCAB)
    norm_pt = np.linalg.norm(product_type_vector)
    if norm_pt > 0: product_type_vector = [x / norm_pt for x in product_type_vector]
    product_type_vector = [x * w_ptype for x in product_type_vector]
    logger.info(f"Applied product_type_weight: {w_ptype}")
    
    availability_query = attributes.get("availability")
    availability_vector = encode_categorical_n_hot(availability_query, AVAILABILITY_VOCAB)
    norm_avail = np.linalg.norm(availability_vector)
    if norm_avail > 0: availability_vector = [x / norm_avail for x in availability_vector]
    availability_vector = [x * w_avail for x in availability_vector]
    logger.info(f"Applied availability_weight: {w_avail}")

    price_target = attributes.get("price_target"); price_min = attributes.get("price_min")
    price_max = attributes.get("price_max"); price_qualitative = attributes.get("price_qualitative")
    query_price_for_encoding: Optional[float] = None
    if price_target is not None: query_price_for_encoding = float(price_target)
    elif price_min is not None and price_max is not None: query_price_for_encoding = (float(price_min) + float(price_max)) / 2.0
    elif price_max is not None: query_price_for_encoding = float(price_max) * 0.8
    elif price_min is not None: query_price_for_encoding = float(price_min) * 1.2
    elif price_qualitative:
        if PRICE_STATS["count"] > 0 :
            if price_qualitative == "cheap": query_price_for_encoding = PRICE_STATS["min"] + (PRICE_STATS["max"] - PRICE_STATS["min"]) * 0.1
            elif price_qualitative == "expensive": query_price_for_encoding = PRICE_STATS["min"] + (PRICE_STATS["max"] - PRICE_STATS["min"]) * 0.9
    price_vector = encode_price_normalized(query_price_for_encoding)
    price_vector = [x * w_price for x in price_vector]
    logger.info(f"Applied price_weight: {w_price}")

    # 4. Concatenate all weighted (and individually normalized) parts
    final_vector_parts = [
        text_segment_vector,
        image_segment_vector,
        brand_vector,
        product_type_vector,
        availability_vector,
        price_vector
    ]
    final_vector: List[float] = [item for sublist in final_vector_parts for item in sublist] # Flatten
    
    # DO NOT L2 Normalize the final_vector here.

    expected_dim = (config.MULTIMODAL_EMBEDDING_DIMENSION * 2) + \
                   len(BRAND_VOCAB) + len(PRODUCT_TYPE_VOCAB) + \
                   len(AVAILABILITY_VOCAB) + 1
    if len(final_vector) != expected_dim:
        logger.error(f"Query vector dim mismatch! Expected {expected_dim}, got {len(final_vector)}. Vocabs loaded: {VOCABS_LOADED_SUCCESSFULLY}")
        return None
    return final_vector

# ... (search_with_llm_parsed_query, format_results, main remain the same)

# Cache for query vectors to improve performance with enhanced security
# Cache size is limited to avoid memory issues
@functools.lru_cache(maxsize=25)  # Further reduced cache size
def get_cached_dense_vector(query_hash: str, parsed_query_json: str, image_path: Optional[str] = None) -> Optional[List[float]]:
    """Cache wrapper for construct_holistic_query_vector with enhanced validation."""
    try:
        # Strict input validation to prevent cache poisoning
        if not isinstance(query_hash, str) or len(query_hash) > 64:
            logger.error("Invalid query hash format")
            return None
            
        if not isinstance(parsed_query_json, str):
            logger.error("Invalid query JSON type")
            return None
            
        # Validate input size to prevent memory issues and DoS
        if len(parsed_query_json) > 5000:  # Stricter limit
            logger.warning(f"Query JSON too large for caching: {len(parsed_query_json)} chars, processing directly")
            try:
                parsed_query = json.loads(parsed_query_json)
                if not isinstance(parsed_query, dict):
                    raise ValueError("Invalid parsed query format")
                return construct_holistic_query_vector(parsed_query, image_path)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in large query: {e}")
                return None
        
        # Validate image path if provided
        if image_path and (not isinstance(image_path, str) or len(image_path) > 1024):
            logger.error("Invalid image path format")
            return None
        
        parsed_query = json.loads(parsed_query_json)
        if not isinstance(parsed_query, dict):
            raise ValueError("Invalid parsed query format")
        
        # Additional validation for required fields
        if "search_text_for_description" not in parsed_query:
            logger.warning("Missing required field in cached query")
            return None
            
        return construct_holistic_query_vector(parsed_query, image_path)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in cached dense vector: {e}")
        return None
    except Exception as e:
        logger.error(f"Error in cached dense vector generation: {e}", exc_info=True)
        return None

@functools.lru_cache(maxsize=25)  # Further reduced cache size
def get_cached_sparse_vector(keyword_text: str) -> Optional[Dict[str, Any]]:
    """Cache wrapper for sparse vector generation with enhanced validation."""
    # Enhanced input validation
    if not keyword_text or not isinstance(keyword_text, str):
        return None
        
    if not sparse_embedding_model_search:
        logger.warning("Sparse embedding model not available")
        return None
    
    # Sanitize input text
    keyword_text = keyword_text.strip()
    if not keyword_text:
        return None
    
    # Validate input size and sanitize
    if len(keyword_text) > 500:  # Stricter limit for keywords
        logger.warning(f"Keyword text too long for caching: {len(keyword_text)} chars, truncating")
        keyword_text = keyword_text[:500]
    
    # Basic sanitization - remove potentially problematic characters
    import re
    keyword_text = re.sub(r'[^\w\s\-\.,]', ' ', keyword_text)
    keyword_text = ' '.join(keyword_text.split())  # Normalize whitespace
    
    if not keyword_text:
        logger.warning("Keyword text became empty after sanitization")
        return None
    
    try:
        sparse_emb_obj_list = list(sparse_embedding_model_search.query_embed(keyword_text))
        if not sparse_emb_obj_list:
            logger.warning(f"No sparse embeddings generated for: {keyword_text[:50]}...")
            return None
            
        sparse_emb_obj = sparse_emb_obj_list[0]
        
        # Validate embedding results
        if not hasattr(sparse_emb_obj, 'indices') or not hasattr(sparse_emb_obj, 'values'):
            logger.error("Invalid sparse embedding object structure")
            return None
            
        indices = sparse_emb_obj.indices.tolist()
        values = sparse_emb_obj.values.tolist()
        
        # Validate embedding dimensions
        if len(indices) != len(values):
            logger.error(f"Sparse embedding dimension mismatch: {len(indices)} indices vs {len(values)} values")
            return None
        
        if len(indices) > 10000:  # Reasonable limit
            logger.warning(f"Sparse embedding too large: {len(indices)} indices")
            return None
        
        return {
            "indices": indices,
            "values": values
        }
    except Exception as e:
        logger.error(f"Error in cached sparse vector generation: {e}", exc_info=True)
        return None

def search_with_llm_parsed_query(
    natural_language_query: str,
    image_path_for_query: Optional[str] = None,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Performs an LLM-enhanced hybrid search for products.

    This is the main search function that orchestrates the following steps:
    1.  Parses the `natural_language_query` using `parse_query_with_llm`.
    2.  Constructs a dense holistic query vector using `construct_holistic_query_vector`
        (utilizing `get_cached_dense_vector` for caching). This vector incorporates
        text, optional image input, and structured attributes.
    3.  Constructs a sparse query vector from keywords extracted by the LLM
        (utilizing `get_cached_sparse_vector` for caching).
    4.  Builds Qdrant filter conditions based on parsed attributes (e.g., brand, price range).
    5.  Calls `vector_store.search_points` to perform a hybrid search in Qdrant
        using both dense and sparse vectors, along with any filters.
    6.  Logs the search duration and results.

    Args:
        natural_language_query: The user's text query.
        image_path_for_query: Optional path to an image file for multimodal search.
        limit: The maximum number of search results to return.

    Returns:
        A list of search result dictionaries, each typically containing 'id', 'score',
        and 'payload' from Qdrant. Returns an empty list if the search fails or yields no results.
    """
    start_time = time.time()
    logger.info(f"Performing LLM-enhanced hybrid search for: '{natural_language_query}'" + (f" with image '{image_path_for_query}'" if image_path_for_query else ""))
    
    parsed_query = parse_query_with_llm(natural_language_query)
    
    # Ensure primary text for dense search is present, even in fallback
    if not parsed_query.get("search_text_for_description"):
        logger.warning("LLM parsing did not yield 'search_text_for_description'. Using raw query for dense search description text.")
        parsed_query["search_text_for_description"] = natural_language_query
        # Ensure attributes and weights are at least empty dicts if LLM failed completely
        if "attributes" not in parsed_query: parsed_query["attributes"] = {}
        if "weights" not in parsed_query: parsed_query["weights"] = {}

    logger.info(f"LLM Parsed Query: {json.dumps(parsed_query, indent=2)}")

    # 1. Construct Dense Holistic Query Vector (using cache)
    # Create a hash of the query and image path for caching
    query_hash = hashlib.md5((str(parsed_query) + str(image_path_for_query)).encode()).hexdigest()
    parsed_query_json = json.dumps(parsed_query, sort_keys=True)
    
    dense_query_vec = get_cached_dense_vector(query_hash, parsed_query_json, image_path_for_query)
    if not dense_query_vec:
        logger.error("Failed to construct dense holistic query vector.")
        # Allow to proceed and try sparse

    # 2. Construct Sparse Query Vector (using cache)
    sparse_query_vector_data = None
    keyword_text_for_sparse = parsed_query.get("keyword_search_text", parsed_query.get("search_text_for_description")) # Fallback to desc text
    
    if keyword_text_for_sparse:
        sparse_query_vector_data = get_cached_sparse_vector(keyword_text_for_sparse)
        if sparse_query_vector_data:
            logger.info(f"Using cached sparse query vector with {len(sparse_query_vector_data['indices'])} indices.")
        else:
            logger.warning(f"Failed to generate sparse query vector for '{keyword_text_for_sparse}'")
    elif not sparse_embedding_model_search:
        logger.warning("Sparse embedding model for search not available. Skipping sparse query part.")

    if not dense_query_vec and not sparse_query_vector_data:
        logger.error("Both dense and sparse query vector construction failed. No search performed.")
        return []

    # 3. Call vector_store.search_points for hybrid search with optimized parameters
    # Add filter conditions based on parsed query attributes
    filter_condition = None
    attributes = parsed_query.get("attributes", {})
    
    # Build filter conditions for more efficient search
    filter_parts = []
    
    # Add brand filter if specified with high confidence
    try:
        if attributes.get("brand") and parsed_query.get("weights", {}).get("brand_weight", 0) > 0.7:
            # Only apply brand filter for exact matches to avoid issues
            # with partial brand names or multi-brand queries
            # Ensure brand is a list, even if single value, for 'should' condition
            brand_values = attributes["brand"]
            if not isinstance(brand_values, list):
                brand_values = [brand_values]
            
            if brand_values: # Proceed only if there are brand values
                brand_conditions = [
                    models.FieldCondition(
                        key="brand",
                        match=models.MatchValue(value=brand_val)
                    ) for brand_val in brand_values
                ]
                # If there's only one brand, use it directly in 'must'
                # If multiple brands were somehow parsed (e.g. "Nike or Adidas"), use 'should'
                if len(brand_conditions) == 1:
                    # Check if the single brand value is simple enough
                    # (no spaces, length > 2, as per original commented logic)
                    # This check was inside the original commented block, applying it here.
                    if " " not in brand_values[0] and len(brand_values[0]) > 2:
                        filter_parts.extend(brand_conditions)
                        logger.info(f"Applied brand filter for: {brand_values[0]}")
                    else:
                        logger.info(f"Skipping brand filter for complex or short single brand value: {brand_values[0]}")
                elif len(brand_conditions) > 1: # Should not happen with current LLM prompt but good for future
                    filter_parts.append(models.Filter(should=brand_conditions))
                    logger.info(f"Applied multi-brand filter for: {brand_values}")
    except Exception as e:
        logger.warning(f"Failed to apply brand filter: {e}")
    
    # Add price range filter if specified
    try:
        if attributes.get("price_min") or attributes.get("price_max"):
            apply_price_filter = True
            if attributes.get("price_max") and float(attributes.get("price_max", 0)) > 10000: # Threshold from previous logic
                apply_price_filter = False
                logger.info(f"Skipping price filter for high price_max: {attributes.get('price_max')}")
            
            if apply_price_filter:
                price_range_filter = {}
                if attributes.get("price_min"):
                    price_range_filter["gte"] = float(attributes["price_min"])
                if attributes.get("price_max"):
                    price_range_filter["lte"] = float(attributes["price_max"])
                
                if price_range_filter: # Ensure there's something to filter on
                    filter_parts.append(
                        models.FieldCondition(
                            key="price_val",
                            range=models.Range(**price_range_filter)
                        )
                    )
                    logger.info(f"Applied price filter: {price_range_filter}")
    except Exception as e:
        logger.warning(f"Failed to apply price filter: {e}")
    
    # Combine filters if any exist
    if filter_parts:
        filter_condition = models.Filter(
            must=filter_parts
        )
    
    results = vector_store.search_points(
        dense_query_vector=dense_query_vec,
        sparse_query_vector_data=sparse_query_vector_data,
        limit=limit,
        filter_condition=filter_condition
    )
    
    search_time = time.time() - start_time
    logger.info(f"Hybrid search found {len(results)} results for query: '{natural_language_query}' in {search_time:.2f} seconds")
    return results

def format_results(results: List[Dict[str, Any]]) -> str:
    """
    Formats a list of search results into a human-readable string.

    Args:
        results: A list of search result dictionaries, as returned by
                 `vector_store.search_points` or `search_with_llm_parsed_query`.
                 Each dictionary is expected to have 'id', 'score', and 'payload'.

    Returns:
        A string representation of the search results, including product details
        like title, brand, price, link, and image. Returns "No results found."
        if the input list is empty.
    """
    if not results: return "No results found."
    formatted = "Search Results:\n\n"
    for i, result in enumerate(results):
        payload = result.get('payload', {})
        formatted += f"Result {i+1} (Score: {result.get('score', 0.0):.4f}):\n"
        formatted += f"  ID (Qdrant): {result.get('id')}\n"
        formatted += f"  Product ID: {payload.get('product_id')}\n"
        formatted += f"  Title: {payload.get('title')}\n"
        formatted += f"  Brand: {payload.get('brand')}\n"
        formatted += f"  Price: {payload.get('price_str')} (Parsed: {payload.get('price_val')})\n"
        formatted += f"  Availability: {payload.get('availability')}\n"
        formatted += f"  Product Type: {payload.get('product_type_full')} (Specific: {payload.get('product_type_specific')})\n"
        formatted += f"  Link: {payload.get('link')}\n"
        formatted += f"  Image: {payload.get('image_link')}\n"
    return formatted

def main():
    """
    Main function to run the search script from the command line.

    Parses command-line arguments for the query, optional image path, and result limit.
    It ensures necessary vocabularies are loaded, then calls `search_with_llm_parsed_query`
    to perform the search, and finally prints the formatted results to the console.
    Handles GEMINI_API_KEY configuration check.
    """
    parser = argparse.ArgumentParser(description='Search for products using LLM-enhanced queries.')
    parser.add_argument('--query', type=str, required=True, help='Natural language text query')
    parser.add_argument('--image', type=str, help='Path to an optional image file for multimodal query')
    parser.add_argument('--limit', type=int, default=5, help='Maximum number of results to return')
    
    args = parser.parse_args()
    
    if not config.GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY is not set in config.py. Please add it to use LLM-enhanced search.")
        print("You can get a key from https://makersuite.google.com/app/apikey")
        return

    # Crucial: Ensure vocabs are loaded/built before any search operation
    ensure_vocabs_for_search() 
    if not VOCABS_LOADED_SUCCESSFULLY:
        print("Could not load or build necessary vocabularies. Aborting search.")
        return

    results = search_with_llm_parsed_query(args.query, image_path_for_query=args.image, limit=args.limit)
    formatted_results = format_results(results)
    print(formatted_results)

if __name__ == "__main__":
    main()