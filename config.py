# config.py

QDRANT_URL = "https://8a8940c5-467a-4bb5-829e-7b3e3613da9f.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.nr2nPtgXc7xVoZ34mjN81XHusFtRAZNu-Edf-sNxV5I"

GOOGLE_CLOUD_PROJECT = "its-koral-prod"
GOOGLE_CLOUD_REGION = "us-central1" # Default region, can be adjusted

# For multimodalembedding@001, the endpoint ID and region might be fixed
# but good to have if we need to specify for other models.
# Typically, the library handles the endpoint for pre-trained models like multimodalembedding.

# Collection name for Qdrant
QDRANT_COLLECTION_NAME = "ai_product_expert_collection"

# Embedding dimensions
MULTIMODAL_EMBEDDING_DIMENSION = 1408
TEXT_EMBEDDING_005_DIMENSION = 768  # Confirmed from testing

# Gemini API Key
# Get this from https://makersuite.google.com/app/apikey
GEMINI_API_KEY = "AIzaSyBzDEIuzmahhu1CSQ4DyTpZveupZcNq398"  # Replace with your actual API key
# Vector Naming for Qdrant Hybrid Search
DEFAULT_DENSE_VECTOR_NAME = "holistic_dense" # Name for our main dense holistic vector
SPARSE_VECTOR_NAME = "minicoil_sparse"   # Name for the MiniCOIL sparse vector
HOLISTIC_DENSE_VECTOR_DIM = 3667 # (1408*2 for text/image) + brand_vocab + type_vocab + avail_vocab + price_scalar