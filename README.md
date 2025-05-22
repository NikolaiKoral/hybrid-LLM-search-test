# AI Product Expert Bot

## Overview

The AI Product Expert Bot is a sophisticated search system designed to provide an advanced product discovery experience. It leverages multimodal understanding (text and images), a hybrid search approach (dense semantic search + sparse keyword search), and Large Language Models (LLMs) to interpret complex user queries and deliver highly relevant product results.

This system is built upon a Retrieval Augmented Generation (RAG) architecture, utilizing Google's multimodal embedding models, Qdrant as a vector database, and Google's Gemini LLM for query understanding and processing.

## Features

-   **Multimodal Search**: Accepts both text and image inputs for queries.
-   **Hybrid Search**: Combines dense vector search for semantic understanding with sparse vector search for keyword precision.
-   **LLM-Powered Query Understanding**: Uses an LLM (Gemini) to parse complex natural language queries, extract attributes, and determine search intent.
-   **Holistic Product Vectors**: Creates comprehensive vector representations for products by concatenating embeddings from various modalities (text, image) and structured attributes (brand, product type, price, availability).
-   **Dynamic Weighting**: Allows for query-time weighting of different vector components based on LLM interpretation of user intent.
-   **Caching**: Implements LRU caching for query vectors to improve performance on repeated searches.
-   **Robust Filtering**: Supports filtering search results by attributes like brand and price range.

## Architecture

The core components of the system are:

1.  **Data Ingestion (`data_ingestion.py`)**:
    *   Parses product data (e.g., from an XML feed).
    *   Generates multimodal embeddings for product text and images using Google's `multimodalembedding@001` model.
    *   Generates sparse embeddings for product text using a MiniCOIL model (e.g., `Qdrant/minicoil-v1`).
    *   Encodes structured attributes (brand, product type, price, availability) into vector segments.
    *   Concatenates these embeddings into a single "holistic" dense vector and also stores the sparse vector.
    *   Stores these vectors and product metadata in a Qdrant collection.

2.  **Qdrant Vector Database (`vector_store.py`, `qdrant_client.py`)**:
    *   Stores and indexes the holistic dense vectors and sparse keyword vectors.
    *   Provides efficient similarity search capabilities.
    *   The collection is configured with named vectors for dense (`holistic_dense`) and sparse (`minicoil_sparse`) representations.

3.  **Search Pipeline (`search.py`)**:
    *   **LLM Query Parsing**: Takes a natural language query (and optionally an image path) and uses an LLM (Gemini) to parse it into:
        *   Search text for description matching.
        *   Search text for image modality matching.
        *   Keywords for sparse search.
        *   Structured attributes (brand, product type, price, availability).
        *   Weights for different components of the dense query vector.
    *   **Query Vector Construction**:
        *   Generates a dense holistic query vector by embedding the parsed text/image components and encoding structured attributes, then applying weights.
        *   Generates a sparse query vector from the extracted keywords.
    *   **Hybrid Search**: Performs searches against Qdrant using both the dense and sparse query vectors.
    *   **Result Fusion**: Combines and re-ranks results from dense and sparse searches (currently a simple max-score approach).
    *   **Caching**: Uses LRU caching for generated dense and sparse query vectors to improve performance.

4.  **Google Cloud Clients (`gcp_clients.py`, `gemini_client.py`, `text_embedding_client.py`)**:
    *   Interfaces with Google Cloud services for multimodal embeddings (Vertex AI) and LLM capabilities (Gemini API).

5.  **Configuration (`config.py`)**:
    *   Manages API keys, model names, Qdrant connection details, and other system parameters.

## Prerequisites

-   Python 3.9+
-   Access to Google Cloud Platform (for Vertex AI and Gemini API).
-   A Qdrant instance (local or cloud).
-   Product data feed (e.g., XML format).

## Setup

1.  **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd ai_product_expert_bot
    ```

2.  **Create a Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file needs to be created. Key dependencies include `qdrant-client`, `google-cloud-aiplatform`, `google-generativeai`, `numpy`, `fastembed`)*

4.  **Configure API Keys and Endpoints**:
    *   Create a `config.py` file (or update the existing one).
    *   Add your Google Cloud Project ID, region, and Gemini API Key:
        ```python
        # config.py
        GCP_PROJECT_ID = "your-gcp-project-id"
        GCP_REGION = "your-gcp-region" # e.g., "us-central1"
        GEMINI_API_KEY = "your-gemini-api-key"
        ```
    *   Configure your Qdrant connection details:
        ```python
        # config.py
        QDRANT_URL = "your_qdrant_url" # e.g., "http://localhost:6333" or cloud URL
        QDRANT_API_KEY = "your_qdrant_api_key" # If applicable
        QDRANT_COLLECTION_NAME = "ai_product_expert_collection"
        ```

5.  **Set up Qdrant**:
    *   Ensure your Qdrant instance is running and accessible.
    *   The collection specified in `QDRANT_COLLECTION_NAME` will be created automatically by the data ingestion script if it doesn't exist, with the correct vector configurations.

## Data Ingestion

1.  Place your product data feed (e.g., `products.xml`) in a suitable location.
2.  Update the `XML_FILE_PATH` in `data_ingestion.py` if necessary.
3.  Run the data ingestion script:
    ```bash
    python3 data_ingestion.py
    ```
    This script will:
    *   Parse the product feed.
    *   Build vocabularies for brands and product types.
    *   Generate and store embeddings for all products in Qdrant.
    *   Save vocabularies and price statistics to `vocab_embeddings.json`.

## Running the Search

Once data ingestion is complete, you can perform searches using the `search.py` script:

**Text-only search:**
```bash
python3 search.py --query "your natural language query here"
```

**Multimodal search (text + image):**
```bash
python3 search.py --query "find products similar to this image with a modern look" --image path/to/your/image.jpg
```

**Optional arguments:**
-   `--limit <N>`: Specify the maximum number of results to return (default is 5).

Example:
```bash
python3 search.py --query "OFYR grill under 10000 DKK"
python3 search.py --query "durable outdoor grill with rustic look" --image images/1020405060118.jpg --limit 3
```

## Future Enhancements

-   Fine-tune embedding models or LLMs for domain-specific improvements.
-   Implement more sophisticated result fusion techniques (e.g., Reciprocal Rank Fusion).
-   Add support for user profiles and personalized search results.
-   Develop a web-based user interface.
-   Expand testing and evaluation metrics.