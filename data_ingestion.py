"""
Data ingestion pipeline for the AI Product Expert Bot.
This script downloads and parses the XML feed, generates embeddings, and stores them in Qdrant.
"""

import os
import logging
import time
import requests
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any, Tuple, Set
from urllib.parse import urlparse
import numpy as np
from PIL import Image
from io import BytesIO
import uuid
import re # For parsing price
import json # For persisting vocab embeddings
import argparse # For command-line arguments
from fastembed import SparseTextEmbedding # Add fastembed

# Import our modules
import config
import vector_store
from gcp_clients import get_multimodal_embeddings
from text_embedding_client import get_text_embedding as get_dedicated_text_embedding # Alias for clarity

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Sparse Embedding Model globally or ensure it's initialized before use
# This might be better in a function that's called once, e.g., in main or ingest_data
# For now, global initialization for simplicity in this step.
try:
    sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/minicoil-v1", batch_size=128)
    logger.info("SparseTextEmbedding model (Qdrant/minicoil-v1) initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize SparseTextEmbedding model: {e}", exc_info=True)
    sparse_embedding_model = None
    # Don't continue if this is critical - consider raising here if required

# XML namespaces
NAMESPACES = {
    'g': 'http://base.google.com/ns/1.0',
    'c': 'http://base.google.com/cns/1.0'
}

# --- Vocabulary, Stats, and Vocab Embeddings ---
BRAND_VOCAB: List[str] = []
PRODUCT_TYPE_VOCAB: List[str] = []
AVAILABILITY_VOCAB: List[str] = ["in_stock", "out_of_stock", "preorder", "backorder"]
PRICE_STATS = {"min": float('inf'), "max": float('-inf'), "count": 0, "sum": 0.0}
PRODUCT_TYPE_VOCAB_EMBEDDINGS: Dict[str, List[float]] = {}

VOCAB_EMBEDDINGS_FILE = "vocab_embeddings.json"

def get_element_text(element: ET.Element, tag_name: str) -> Optional[str]:
    """Gets the text content of an element with the given tag name."""
    if ':' in tag_name:
        namespace, local_name = tag_name.split(':')
        tag = f"{{{NAMESPACES[namespace]}}}{local_name}"
        elem = element.find(tag)
    else:
        elem = element.find(tag_name)
    return elem.text.strip() if elem is not None and elem.text else None

def parse_price(price_str: Optional[str]) -> Optional[float]:
    """Parses price string (e.g., "189.95 DKK") to float."""
    if not price_str: return None
    match = re.search(r'[\d\.]+', price_str)
    if match:
        try: return float(match.group(0))
        except ValueError: return None
    return None

def load_persisted_vocab_embeddings():
    """Loads vocabularies and their embeddings from a file if it exists."""
    global BRAND_VOCAB, PRODUCT_TYPE_VOCAB, PRICE_STATS, PRODUCT_TYPE_VOCAB_EMBEDDINGS
    if os.path.exists(VOCAB_EMBEDDINGS_FILE):
        try:
            with open(VOCAB_EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate data structure
            if not isinstance(data, dict):
                raise ValueError("Invalid vocab embeddings file format")
            
            BRAND_VOCAB = data.get("BRAND_VOCAB", [])
            PRODUCT_TYPE_VOCAB = data.get("PRODUCT_TYPE_VOCAB", [])
            PRICE_STATS = data.get("PRICE_STATS", PRICE_STATS)
            PRODUCT_TYPE_VOCAB_EMBEDDINGS = data.get("PRODUCT_TYPE_VOCAB_EMBEDDINGS", {})
            
            # Validate loaded data
            if not isinstance(BRAND_VOCAB, list) or not isinstance(PRODUCT_TYPE_VOCAB, list):
                raise ValueError("Invalid vocabulary format")
            
            logger.info(f"Loaded vocabularies and embeddings from {VOCAB_EMBEDDINGS_FILE}")
            logger.info(f"Brands: {len(BRAND_VOCAB)}, Types: {len(PRODUCT_TYPE_VOCAB)}, Type Embeddings: {len(PRODUCT_TYPE_VOCAB_EMBEDDINGS)}")
            return True
        except Exception as e:
            logger.error(f"Error loading {VOCAB_EMBEDDINGS_FILE}: {e}. Will rebuild.")
    return False

def persist_vocab_embeddings():
    """Saves vocabularies and their embeddings to a file atomically."""
    data_to_persist = {
        "BRAND_VOCAB": BRAND_VOCAB,
        "PRODUCT_TYPE_VOCAB": PRODUCT_TYPE_VOCAB,
        "PRICE_STATS": PRICE_STATS,
        "PRODUCT_TYPE_VOCAB_EMBEDDINGS": PRODUCT_TYPE_VOCAB_EMBEDDINGS
    }
    
    # Write to temporary file first, then rename for atomic operation
    temp_file = f"{VOCAB_EMBEDDINGS_FILE}.tmp"
    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data_to_persist, f, indent=2, ensure_ascii=False)
        
        # Atomic rename
        os.replace(temp_file, VOCAB_EMBEDDINGS_FILE)
        logger.info(f"Successfully persisted vocabularies and embeddings to {VOCAB_EMBEDDINGS_FILE}")
    except Exception as e:
        logger.error(f"Error persisting vocab data: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass


def build_vocabularies_and_stats(products: List[Dict[str, Any]], force_rebuild: bool = False):
    """Scans products to build vocabularies, stats, and embeddings for product types."""
    global BRAND_VOCAB, PRODUCT_TYPE_VOCAB, PRICE_STATS, PRODUCT_TYPE_VOCAB_EMBEDDINGS

    if not force_rebuild and load_persisted_vocab_embeddings():
        # If loaded successfully and not forcing rebuild, we are done.
        # Ensure all necessary components were loaded.
        if BRAND_VOCAB and PRODUCT_TYPE_VOCAB and PRICE_STATS["count"] > 0 and PRODUCT_TYPE_VOCAB_EMBEDDINGS:
             logger.info("Using successfully loaded persisted vocabularies and embeddings.")
             return
        else:
            logger.warning("Persisted data was incomplete. Rebuilding vocabularies and embeddings.")

    logger.info("Building vocabularies, statistics, and product type embeddings from product data...")

    brands: Set[str] = set()
    product_types: Set[str] = set()
    prices: List[float] = []

    for product in products:
        brand = product.get('brand')
        if brand: brands.add(brand)
        ptype_full = product.get('product_type')
        if ptype_full:
            specific_ptype = ptype_full.split('>')[-1].strip()
            if specific_ptype: product_types.add(specific_ptype)
        price_val = parse_price(product.get('price'))
        if price_val is not None: prices.append(price_val)

    BRAND_VOCAB = sorted(list(brands))
    PRODUCT_TYPE_VOCAB = sorted(list(product_types))
    PRODUCT_TYPE_VOCAB_EMBEDDINGS.clear() # Clear old embeddings before rebuilding

    if prices:
        PRICE_STATS["min"] = min(prices)
        PRICE_STATS["max"] = max(prices)
        PRICE_STATS["count"] = len(prices)
        PRICE_STATS["sum"] = sum(prices)
    else:
        PRICE_STATS.update({"min": 0.0, "max": 1.0, "count": 0, "sum": 0.0})

    logger.info(f"Found {len(BRAND_VOCAB)} unique brands.")
    logger.info(f"Found {len(PRODUCT_TYPE_VOCAB)} unique specific product types. Now generating their embeddings...")

    # Generate embeddings for product type vocabulary
    # Batching for efficiency if PRODUCT_TYPE_VOCAB is large
    batch_size = 50 # Text embedding API might have batch limits
    for i in range(0, len(PRODUCT_TYPE_VOCAB), batch_size):
        batch_terms = PRODUCT_TYPE_VOCAB[i:i+batch_size]
        try:
            embeddings_list = get_dedicated_text_embedding(batch_terms) # Expects list, returns list of lists
            if embeddings_list and len(embeddings_list) == len(batch_terms):
                for term, emb_vector in zip(batch_terms, embeddings_list):
                    PRODUCT_TYPE_VOCAB_EMBEDDINGS[term] = emb_vector
            else:
                logger.error(f"Mismatch in returned embeddings for batch {i//batch_size}. Expected {len(batch_terms)}, got {len(embeddings_list) if embeddings_list else 0}")
                # Fallback: embed one by one for this batch
                for term in batch_terms:
                    emb_vector = get_dedicated_text_embedding(term) # Single term
                    if emb_vector:
                        PRODUCT_TYPE_VOCAB_EMBEDDINGS[term] = emb_vector
        except Exception as e:
            logger.error(f"Error embedding product type batch: {e}. Trying one by one.")
            for term in batch_terms: # Fallback for individual errors
                emb_vector = get_dedicated_text_embedding(term)
                if emb_vector: PRODUCT_TYPE_VOCAB_EMBEDDINGS[term] = emb_vector
        logger.info(f"Embedded {len(PRODUCT_TYPE_VOCAB_EMBEDDINGS)} / {len(PRODUCT_TYPE_VOCAB)} product types...")


    logger.info(f"Price stats: Min={PRICE_STATS['min']}, Max={PRICE_STATS['max']}")
    logger.info(f"Generated embeddings for {len(PRODUCT_TYPE_VOCAB_EMBEDDINGS)} product types.")
    
    persist_vocab_embeddings()


# --- Encoding Functions ---
def encode_categorical_n_hot(value: Optional[str], vocab: List[str]) -> List[float]:
    encoding = [0.0] * len(vocab)
    if value and value in vocab:
        try: encoding[vocab.index(value)] = 1.0
        except ValueError: pass
    return encoding

def encode_price_normalized(price_val: Optional[float]) -> List[float]:
    if price_val is None: return [0.5]
    min_p, max_p = PRICE_STATS["min"], PRICE_STATS["max"]
    if max_p == min_p: return [0.5]
    normalized_price = (price_val - min_p) / (max_p - min_p)
    return [max(0.0, min(1.0, normalized_price))]


def download_xml_feed(url: str, max_retries: int = 3) -> Optional[str]:
    """Download XML feed with retry logic and validation."""
    from urllib.parse import urlparse
    
    # Validate URL
    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ['http', 'https']:
            raise ValueError(f"Invalid URL scheme: {parsed_url.scheme}")
    except Exception as e:
        logger.error(f"Invalid URL {url}: {e}")
        return None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading XML feed from {url} (attempt {attempt + 1}/{max_retries})")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            # Validate content type
            content_type = response.headers.get('content-type', '').lower()
            if 'xml' not in content_type and 'text' not in content_type:
                logger.warning(f"Unexpected content type: {content_type}")
            
            logger.info(f"Successfully downloaded XML feed ({len(response.text)} bytes)")
            return response.text
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                logger.error(f"All {max_retries} attempts failed")
                return None
            time.sleep(2 ** attempt)  # Exponential backoff
    return None

def parse_xml_feed(xml_content: str) -> List[Dict[str, Any]]:
    try:
        logger.info("Parsing XML feed")
        root = ET.fromstring(xml_content)
        items = root.findall('.//item')
        logger.info(f"Found {len(items)} products in the feed")
        products = []
        for item in items:
            product = {
                'title': get_element_text(item, 'title'),
                'description': get_element_text(item, 'description'),
                'link': get_element_text(item, 'link'),
                'id': get_element_text(item, 'g:id'),
                'brand': get_element_text(item, 'g:brand'),
                'price': get_element_text(item, 'g:price'),
                'image_link': get_element_text(item, 'g:image_link'),
                'availability': get_element_text(item, 'g:availability'),
                'product_type': get_element_text(item, 'g:product_type'),
                'gtin': get_element_text(item, 'g:gtin'),
                'google_product_category': get_element_text(item, 'g:google_product_category')
            }
            products.append(product)
        logger.info(f"Successfully parsed {len(products)} products")
        return products
    except Exception as e:
        logger.error(f"Failed to parse XML feed: {e}")
        return []

def download_image(url: str, max_size_mb: int = 10) -> Optional[Image.Image]:
    """Download and validate image with security checks."""
    from urllib.parse import urlparse
    
    try:
        # Validate URL
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ['http', 'https']:
            logger.warning(f"Invalid URL scheme for image: {parsed_url.scheme}")
            return None
        
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        # Check content length
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > max_size_mb * 1024 * 1024:
            logger.warning(f"Image too large: {content_length} bytes (max: {max_size_mb}MB)")
            return None
        
        # Validate content type
        content_type = response.headers.get('content-type', '').lower()
        if not content_type.startswith('image/'):
            logger.warning(f"Invalid content type for image: {content_type}")
            return None
        
        # Read and validate image
        img_data = response.content
        if len(img_data) > max_size_mb * 1024 * 1024:
            logger.warning(f"Image data too large: {len(img_data)} bytes")
            return None
        
        img = Image.open(BytesIO(img_data))
        img.verify()  # Validate image structure
        img = Image.open(BytesIO(img_data))  # Reopen after verify
        
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            img = img.convert('RGB')
        return img
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {e}")
        return None

def save_image_locally(img: Image.Image, product_id: str) -> Optional[str]:
    try:
        os.makedirs('images', exist_ok=True)
        path = f"images/{product_id}.jpg"
        img.save(path, "JPEG")
        return path
    except Exception as e:
        logger.error(f"Failed to save image for product {product_id}: {e}")
        return None

def process_product(product: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        product_id = product['id']
        if not product_id:
            logger.warning(f"Product missing ID, skipping: {product.get('title', 'N/A')}")
            return None
        # logger.info(f"Processing product {product_id}: {product['title']}") # Reduce verbosity

        MAX_TEXT_LENGTH = 900
        text_content = product['title'] or ""
        if product.get('description'):
            if len(text_content) < MAX_TEXT_LENGTH :
                text_content += ". "
                remaining_chars = MAX_TEXT_LENGTH - len(text_content)
                if remaining_chars > 0:
                    text_content += product['description'][:remaining_chars]
                    if len(product['description']) > remaining_chars:
                        text_content += "..."
        if len(text_content) > 1024: text_content = text_content[:1024]
        # logger.info(f"Prepared text for embedding: {len(text_content)} characters")

        image_path = None
        if product['image_link']:
            img = download_image(product['image_link'])
            if img: image_path = save_image_locally(img, product_id)

        image_embedding_list, text_embedding_list = get_multimodal_embeddings(
            image_path=image_path, contextual_text=text_content if text_content else None
        )

        brand_vector = encode_categorical_n_hot(product.get('brand'), BRAND_VOCAB)
        specific_ptype = None
        ptype_full = product.get('product_type')
        if ptype_full: specific_ptype = ptype_full.split('>')[-1].strip()
        product_type_vector = encode_categorical_n_hot(specific_ptype, PRODUCT_TYPE_VOCAB)
        availability_vector = encode_categorical_n_hot(product.get('availability'), AVAILABILITY_VOCAB)
        price_val = parse_price(product.get('price'))
        price_vector = encode_price_normalized(price_val) # This is already a list e.g. [0.5]

        # Prepare text and image embedding segments separately
        text_segment_vector = [0.0] * config.MULTIMODAL_EMBEDDING_DIMENSION
        if text_embedding_list:
            norm_text = np.linalg.norm(text_embedding_list)
            if norm_text > 0:
                text_segment_vector = [x / norm_text for x in text_embedding_list]
        
        image_segment_vector = [0.0] * config.MULTIMODAL_EMBEDDING_DIMENSION
        if image_embedding_list:
            norm_image = np.linalg.norm(image_embedding_list)
            if norm_image > 0:
                image_segment_vector = [x / norm_image for x in image_embedding_list]

        # Ensure at least one of text or image embedding was successful if no structured data is compelling enough
        if not text_embedding_list and not image_embedding_list:
            logger.warning(f"No text or image embedding for product {product_id}. Skipping if other parts are also weak.")
            # Depending on strategy, might still proceed if structured data is rich.
            # For now, if both are missing, it's a critical failure for the multimodal aspect.
            # Consider if a product with only structured data should be indexed.
            # For this iteration, let's require at least one multimodal component.
            # The check below for `any(final_vector)` might catch this, but this is more explicit.
            # However, the current `get_multimodal_embeddings` returns None, None if both inputs are None.
            # And `process_product` already checks if `image_path` and `text_content` are both None.
            # The main concern is if `get_multimodal_embeddings` *fails* for one part.
            # The current `get_multimodal_embeddings` returns (None, text_emb) or (img_emb, None) if one part fails or is missing.
            # So, text_embedding_list or image_embedding_list could be None.

        final_vector_parts = [
            text_segment_vector,    # Segment 1: Text embedding from multimodal model
            image_segment_vector,   # Segment 2: Image embedding from multimodal model
            brand_vector,           # Segment 3
            product_type_vector,    # Segment 4
            availability_vector,    # Segment 5
            price_vector            # Segment 6
        ]
        final_vector: List[float] = [item for sublist in final_vector_parts for item in sublist] # Flatten

        # Log sum of absolute values to check magnitude
        sum_abs_final_vector = sum(abs(x) for x in final_vector)
        is_near_zero = not any(abs(x) > 1e-9 for x in final_vector)
        logger.info(f"Product {product_id}: Dense vector sum_abs={sum_abs_final_vector}, is_near_zero={is_near_zero}. First 5 elements: {final_vector[:5]}") # Changed to info

        # Check if the vector is all zeros (e.g., if all parts failed or were empty)
        # This is important because Qdrant might not accept all-zero vectors depending on config or version.
        if is_near_zero:
             logger.warning(f"Generated an all-zero or near-zero (dense) vector for product {product_id} (sum_abs={sum_abs_final_vector}). Skipping.")
             # Log parts for debugging - changed to INFO for visibility
             logger.info(f"Product {product_id} - text_segment_vector sum_abs: {sum(abs(x) for x in text_segment_vector)}")
             logger.info(f"Product {product_id} - image_segment_vector sum_abs: {sum(abs(x) for x in image_segment_vector)}")
             logger.info(f"Product {product_id} - brand_vector sum_abs: {sum(abs(x) for x in brand_vector)}")
             logger.info(f"Product {product_id} - product_type_vector sum_abs: {sum(abs(x) for x in product_type_vector)}")
             logger.info(f"Product {product_id} - availability_vector sum_abs: {sum(abs(x) for x in availability_vector)}")
             logger.info(f"Product {product_id} - price_vector sum_abs: {sum(abs(x) for x in price_vector)}")
             return None

        # Generate Sparse Embedding for text (title + description)
        sparse_text_for_minicoil = (product.get('title') or "") + " " + (product.get('description') or "")
        sparse_text_for_minicoil = sparse_text_for_minicoil.strip()
        
        product_sparse_vector_data = None
        if sparse_text_for_minicoil and sparse_embedding_model:
            try:
                # FastEmbed's embed method for SparseTextEmbedding returns a generator of SparseEmbedding objects
                # We expect one document, so take the first.
                sparse_embeddings_gen = sparse_embedding_model.embed([sparse_text_for_minicoil])
                sparse_embedding_result = next(sparse_embeddings_gen, None)
                if sparse_embedding_result:
                    product_sparse_vector_data = {
                        "indices": sparse_embedding_result.indices.tolist(), # Ensure they are lists
                        "values": sparse_embedding_result.values.tolist()
                    }
                else:
                    logger.warning(f"Sparse embedding generation returned None for product {product_id}")
            except Exception as e:
                logger.error(f"Error generating sparse embedding for product {product_id}: {e}", exc_info=True)
        elif not sparse_embedding_model:
            logger.warning("Sparse embedding model not available, skipping sparse vector generation.")


        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, product_id))
        
        # Structure for vector_store.upsert_points
        # It expects 'dense_vector' and 'sparse_vector' keys in the point_data dictionary
        # The 'vector' key in PointStruct will be a dict mapping names to vectors.
        # So, process_product should return a dict that upsert_points can easily use.
        
        # The `upsert_points` in vector_store.py now expects a dictionary like:
        # {'id': ..., 'payload': ..., 'dense_vector': ..., 'sparse_vector': {'indices': ..., 'values': ...}}
        
        return_data = {
            'id': point_id,
            'dense_vector': final_vector, # This is the concatenated holistic dense vector
            'sparse_vector': product_sparse_vector_data, # This is {'indices': ..., 'values': ...} or None
            'payload': {
                'product_id': product_id, 'title': product.get('title'),
                'description': product.get('description'), 'brand': product.get('brand'),
                'price_str': product.get('price'), 'price_val': price_val,
                'image_link': product.get('image_link'), 'local_image_path': image_path,
                'availability': product.get('availability'),
                'product_type_full': product.get('product_type'),
                'product_type_specific': specific_ptype, 'link': product.get('link'),
                'gtin': product.get('gtin'),
                'google_product_category': product.get('google_product_category'),
            }
        }
        logger.info(f"Product {product_id}: Successfully prepared return_data. Dense sum_abs={sum_abs_final_vector}.") # Changed to info and added sum_abs
        return return_data
    except Exception as e:
        logger.error(f"Failed to process product {product.get('id', 'unknown')}: {e}", exc_info=True)
        return None

def ingest_data(feed_url: str, batch_size: int = 10, limit: Optional[int] = None, force_rebuild_vocabs: bool = False) -> bool:
    try:
        xml_content = download_xml_feed(feed_url)
        if not xml_content: return False
        products = parse_xml_feed(xml_content)
        if not products: return False

        build_vocabularies_and_stats(products, force_rebuild=force_rebuild_vocabs)
        
        new_vector_size = (config.MULTIMODAL_EMBEDDING_DIMENSION * 2) + \
                          len(BRAND_VOCAB) + \
                          len(PRODUCT_TYPE_VOCAB) + \
                          len(AVAILABILITY_VOCAB) + \
                          1 # for price (text_emb + image_emb + brand + type + avail + price)
        logger.info(f"Calculated new holistic vector size: {new_vector_size}")

        # Get current collection name from config
        collection_name = config.QDRANT_COLLECTION_NAME
        
        # Check existing collection and decide if recreation is needed
        current_collection_info = vector_store.get_collection_info(collection_name)
        recreate_collection_flag = False

        if not current_collection_info:
            logger.info(f"Collection '{collection_name}' does not exist. It will be created with hybrid configuration.")
            # No need to set recreate_collection_flag = True, ensure_collection_exists will create it.
        else:
            # Check if the existing collection is compatible with the new hybrid schema.
            # The new `get_collection_info` returns `vector_size_info` like "holistic_dense: 3667, minicoil_sparse: Sparse"
            # An old schema might just have a number or be missing the sparse part.
            vector_size_info_str = current_collection_info.get('vector_size_info', '')
            
            expected_dense_name = config.DEFAULT_DENSE_VECTOR_NAME
            expected_sparse_name = config.SPARSE_VECTOR_NAME
            expected_dense_dim_str = str(config.HOLISTIC_DENSE_VECTOR_DIM)

            # Check for presence of named vectors and correct dense dimension
            # This is a heuristic check. A more robust check would involve parsing vector_size_info_str precisely.
            if not (expected_dense_name in vector_size_info_str and \
                    expected_sparse_name in vector_size_info_str and \
                    f"{expected_dense_name}: {expected_dense_dim_str}" in vector_size_info_str):
                logger.warning(
                    f"Existing collection '{collection_name}' configuration mismatch or doesn't support hybrid setup. "
                    f"Expected dense '{expected_dense_name}: {expected_dense_dim_str}' and sparse '{expected_sparse_name}'. "
                    f"Found: '{vector_size_info_str}'. Recreating collection."
                )
                recreate_collection_flag = True
            else:
                logger.info(f"Existing collection '{collection_name}' appears compatible with hybrid setup.")

        if recreate_collection_flag:
            logger.info(f"Deleting existing incompatible collection '{collection_name}' before creating new one.")
            vector_store.delete_collection(collection_name)
            # After deletion, ensure_collection_exists will create it fresh.

        # Ensure collection exists with the correct hybrid configuration (dense + sparse)
        # This call uses defaults from config.py for vector names and dense dimension.
        if not vector_store.ensure_collection_exists(collection_name=collection_name):
            logger.error(f"Failed to ensure Qdrant collection '{collection_name}' exists with the new hybrid configuration.")
            return False
        
        # Verify again after ensure_collection_exists (optional, but good for sanity)
        final_collection_info = vector_store.get_collection_info(collection_name)
        if not final_collection_info:
             logger.error(f"Critical: Collection '{collection_name}' not found after ensure_collection_exists call.")
             return False
        logger.info(f"Successfully ensured collection '{collection_name}' is ready. Config: {final_collection_info.get('vector_size_info')}")

        if limit and limit > 0:
            products_to_ingest = products[:limit]
            logger.info(f"Limited to processing {len(products_to_ingest)} products for ingestion")
        else:
            products_to_ingest = products
            
        total_products_to_ingest = len(products_to_ingest)
        total_processed_count = 0
        total_success_count = 0
        
        for i in range(0, total_products_to_ingest, batch_size):
            batch = products_to_ingest[i:i+batch_size]
            # logger.info(f"Processing batch {i//batch_size + 1}/{(total_products_to_ingest-1)//batch_size + 1} ({len(batch)} products)")
            
            processed_products_batch = []
            for product in batch:
                result = process_product(product)
                if result:
                    processed_products_batch.append(result)
                    total_success_count += 1
                total_processed_count += 1
            
            if processed_products_batch:
                upsert_success = vector_store.upsert_points(processed_products_batch)
                if not upsert_success: logger.error(f"Failed to upsert batch {i//batch_size + 1}")
            
            logger.info(f"Progress: {total_processed_count}/{total_products_to_ingest} products processed, {total_success_count} successful.")
            if i + batch_size < total_products_to_ingest: time.sleep(1) # Avoid overwhelming APIs if not the last batch
        
        logger.info(f"Data ingestion completed. {total_success_count}/{total_products_to_ingest} products successfully processed.")
        return True
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}", exc_info=True)
        return False

def main():
    parser = argparse.ArgumentParser(description="Data ingestion pipeline for AI Product Expert Bot.")
    parser.add_argument("--limit", type=int, default=20, help="Number of products to ingest (default: 20). Use 0 for all.")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for processing (default: 10).")
    parser.add_argument("--force_rebuild_vocabs", action="store_true", help="Force rebuild of vocabularies and embeddings even if persisted file exists.")
    args = parser.parse_args()

    limit_val = args.limit if args.limit > 0 else None

    logger.info("Starting data ingestion pipeline (Phase 2 - Structured Attributes)")
    feed_url = "https://files.channable.com/gOBn8ftRlhqTvgdOkoBnIw==.xml"
    success = ingest_data(feed_url, batch_size=args.batch_size, limit=limit_val, force_rebuild_vocabs=args.force_rebuild_vocabs)
    
    if success:
        logger.info("Data ingestion pipeline completed successfully")
    else:
        logger.error("Data ingestion pipeline failed")

if __name__ == "__main__":
    main()