"""
Text embedding client for the AI Product Expert Bot.
"""

import logging
from typing import List, Optional, Union, Dict, Any

from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
import numpy as np

# Import config
try:
    import config
except ModuleNotFoundError:
    print("ERROR: config.py not found. Please ensure it's in the same directory or PYTHONPATH is set.")
    # Fallback for critical configs if needed
    config = type('obj', (object,), {
        'GOOGLE_CLOUD_PROJECT': 'your-gcp-project-id-fallback',
        'GOOGLE_CLOUD_REGION': 'us-central1',
    })

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Vertex AI
try:
    aiplatform.init(project=config.GOOGLE_CLOUD_PROJECT, location=config.GOOGLE_CLOUD_REGION)
    logger.info(f"Vertex AI initialized for project: {config.GOOGLE_CLOUD_PROJECT} in region: {config.GOOGLE_CLOUD_REGION}")
except Exception as e:
    logger.error(f"Failed to initialize Vertex AI: {e}")

_text_embedding_client = None
_text_embedding_model_name = "text-embedding-005"

def get_text_embedding_client():
    """
    Returns a singleton instance of the text embedding client.
    
    Returns:
        The text embedding client instance.
    """
    global _text_embedding_client
    if _text_embedding_client is None:
        try:
            logger.info("Initializing text embedding client")
            _text_embedding_client = TextEmbeddingModel.from_pretrained(_text_embedding_model_name)
            logger.info(f"Text embedding client initialized successfully for model: {_text_embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize text embedding client: {e}")
            _text_embedding_client = None
    return _text_embedding_client

def get_text_embedding(
    text: Union[str, List[str]],
    normalize: bool = True
) -> Optional[Union[List[float], List[List[float]]]]:
    """
    Generates text embeddings for the given text.
    
    Args:
        text: The text to embed. Can be a single string or a list of strings.
        normalize: Whether to L2-normalize the embeddings.
        
    Returns:
        The text embeddings as a list of floats (for a single text) or a list of lists of floats (for multiple texts).
        Returns None if an error occurs.
    """
    client = get_text_embedding_client()
    if not client:
        logger.error("Text embedding client is not available.")
        return None
    
    try:
        # Convert single string to list for consistent handling
        if isinstance(text, str):
            text_list = [text]
            single_input = True
        else:
            text_list = text
            single_input = False
        
        logger.info(f"Generating embeddings for {len(text_list)} text(s)")
        
        # Get embeddings
        result = []
        for t in text_list:
            # The new API processes one text at a time
            embedding = client.get_embeddings([t])
            values = embedding[0].values
            
            # Normalize if requested
            if normalize:
                norm = np.linalg.norm(values)
                if norm > 0:
                    values = [v / norm for v in values]
            
            result.append(values)
        
        logger.info(f"Successfully generated embeddings with dimension: {len(result[0])}")
        
        # Return single embedding or list of embeddings based on input
        if single_input:
            return result[0]
        else:
            return result
    
    except Exception as e:
        logger.error(f"Failed to generate text embeddings: {e}")
        return None

if __name__ == "__main__":
    # Test the text embedding client
    logger.info("Testing text embedding client...")
    
    # Test single text embedding
    test_text = "This is a test sentence for embedding."
    logger.info(f"Generating embedding for: '{test_text}'")
    
    embedding = get_text_embedding(test_text)
    if embedding:
        logger.info(f"Successfully generated embedding with dimension: {len(embedding)}")
        # Print first few values
        logger.info(f"First few values: {embedding[:5]}")
    else:
        logger.error("Failed to generate embedding.")
    
    # Test batch text embedding
    test_texts = [
        "This is the first test sentence.",
        "This is the second test sentence.",
        "This is the third test sentence."
    ]
    logger.info(f"Generating embeddings for {len(test_texts)} texts")
    
    embeddings = get_text_embedding(test_texts)
    if embeddings:
        logger.info(f"Successfully generated {len(embeddings)} embeddings with dimension: {len(embeddings[0])}")
    else:
        logger.error("Failed to generate batch embeddings.")
    
    logger.info("Text embedding client tests completed.")