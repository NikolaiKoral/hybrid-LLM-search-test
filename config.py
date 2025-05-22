# config.py

import os
from typing import Optional

# Critical Security Fix: Use environment variables for sensitive data
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_REGION = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")

# For multimodalembedding@001, the endpoint ID and region might be fixed
# but good to have if we need to specify for other models.
# Typically, the library handles the endpoint for pre-trained models like multimodalembedding.

# Collection name for Qdrant
QDRANT_COLLECTION_NAME = "ai_product_expert_collection"

# Embedding dimensions
MULTIMODAL_EMBEDDING_DIMENSION = 1408
TEXT_EMBEDDING_005_DIMENSION = 768  # Confirmed from testing

# Gemini API Key - Get this from https://makersuite.google.com/app/apikey
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def validate_config():
    """Validate that all required environment variables are set."""
    missing = []
    if not QDRANT_API_KEY:
        missing.append("QDRANT_API_KEY")
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if not GOOGLE_CLOUD_PROJECT:
        missing.append("GOOGLE_CLOUD_PROJECT")
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Please set these environment variables before running the application."
        )
    
    # Validate configuration values
    if QDRANT_URL and not (QDRANT_URL.startswith('http://') or QDRANT_URL.startswith('https://')):
        raise ValueError("QDRANT_URL must be a valid HTTP/HTTPS URL")
    
    # Validate vector dimensions are positive integers
    if MULTIMODAL_EMBEDDING_DIMENSION <= 0:
        raise ValueError("MULTIMODAL_EMBEDDING_DIMENSION must be positive")
    if TEXT_EMBEDDING_005_DIMENSION <= 0:
        raise ValueError("TEXT_EMBEDDING_005_DIMENSION must be positive")
    if HOLISTIC_DENSE_VECTOR_DIM <= 0:
        raise ValueError("HOLISTIC_DENSE_VECTOR_DIM must be positive")
# Vector Naming for Qdrant Hybrid Search
DEFAULT_DENSE_VECTOR_NAME = "holistic_dense" # Name for our main dense holistic vector
SPARSE_VECTOR_NAME = "minicoil_sparse"   # Name for the MiniCOIL sparse vector
HOLISTIC_DENSE_VECTOR_DIM = 3667 # (1408*2 for text/image) + brand_vocab + type_vocab + avail_vocab + price_scalar