import logging
import os
from typing import Optional, Tuple, List

from google.cloud import aiplatform
from google.cloud.aiplatform_v1beta1.types import PredictResponse # For type hinting
from vertexai.preview.vision_models import Image, MultiModalEmbeddingModel

# Assuming config.py is in the same directory or Python path is set up
try:
    import config
except ModuleNotFoundError:
    print("ERROR: config.py not found. Please ensure it's in the same directory or PYTHONPATH is set.")
    # Fallback for critical configs if needed, or raise error
    config = type('obj', (object,), {
        'GOOGLE_CLOUD_PROJECT': 'your-gcp-project-id-fallback',
        'GOOGLE_CLOUD_REGION': 'us-central1',
        'MULTIMODAL_EMBEDDING_DIMENSION': 1408
    })


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Vertex AI
try:
    aiplatform.init(project=config.GOOGLE_CLOUD_PROJECT, location=config.GOOGLE_CLOUD_REGION)
    logger.info(f"Vertex AI initialized for project: {config.GOOGLE_CLOUD_PROJECT} in region: {config.GOOGLE_CLOUD_REGION}")
except Exception as e:
    logger.error(f"Failed to initialize Vertex AI: {e}")
    # Potentially raise the error or handle it to prevent app crash if init is critical at import time

_multimodal_embedding_model = None

def get_multimodal_embedding_model() -> Optional[MultiModalEmbeddingModel]:
    """Initializes and returns the multimodal embedding model."""
    global _multimodal_embedding_model
    if _multimodal_embedding_model is None:
        try:
            logger.info("Loading multimodal embedding model: multimodalembedding@001")
            _multimodal_embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
            logger.info("Multimodal embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load multimodal embedding model: {e}")
            _multimodal_embedding_model = None # Ensure it stays None on failure
    return _multimodal_embedding_model

def get_multimodal_embeddings(
    image_path: Optional[str] = None,
    gcs_image_uri: Optional[str] = None,
    contextual_text: Optional[str] = None
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Generates multimodal embeddings for a given image and/or text.

    Args:
        image_path: Local path to the image file.
        gcs_image_uri: Google Cloud Storage URI of the image.
        contextual_text: Text to embed.

    Returns:
        A tuple containing (image_embedding, text_embedding).
        Each embedding is a list of floats or None if not generated.
    """
    model = get_multimodal_embedding_model()
    if not model:
        logger.error("Multimodal embedding model is not available.")
        return None, None

    if not image_path and not gcs_image_uri and not contextual_text:
        logger.warning("No input provided for multimodal embeddings (image or text).")
        return None, None
    
    # Input validation
    if contextual_text and len(contextual_text.strip()) == 0:
        logger.warning("Empty text provided for embedding.")
        contextual_text = None
    
    if contextual_text and len(contextual_text) > 10000:
        logger.warning(f"Text too long ({len(contextual_text)} chars), truncating to 10000 chars.")
        contextual_text = contextual_text[:10000]

    image_input: Optional[Image] = None
    if image_path:
        # Validate image path
        if not os.path.exists(image_path):
            logger.error(f"Image file does not exist: {image_path}")
            return None, None
        
        if not os.path.isfile(image_path):
            logger.error(f"Image path is not a file: {image_path}")
            return None, None
        
        # Check file size (max 10MB)
        try:
            file_size = os.path.getsize(image_path)
            if file_size > 10 * 1024 * 1024:  # 10MB
                logger.error(f"Image file too large: {file_size} bytes (max 10MB)")
                return None, None
        except OSError as e:
            logger.error(f"Cannot get file size for {image_path}: {e}")
            return None, None
        
        try:
            image_input = Image.load_from_file(image_path)
            logger.info(f"Loaded image from local path: {image_path}")
        except Exception as e:
            logger.error(f"Failed to load image from path {image_path}: {e}")
            return None, None
    elif gcs_image_uri:
        # Validate GCS URI format
        if not gcs_image_uri.startswith('gs://'):
            logger.error(f"Invalid GCS URI format: {gcs_image_uri}")
            return None, None
        
        try:
            image_input = Image(gcs_uri=gcs_image_uri)
            logger.info(f"Using image from GCS URI: {gcs_image_uri}")
        except Exception as e:
            logger.error(f"Failed to load image from GCS URI {gcs_image_uri}: {e}")
            return None, None


    try:
        logger.info(f"Requesting embeddings with dimension: {config.MULTIMODAL_EMBEDDING_DIMENSION}")
        embeddings = model.get_embeddings(
            image=image_input,
            contextual_text=contextual_text,
            dimension=config.MULTIMODAL_EMBEDDING_DIMENSION,
        )
        
        image_embedding = embeddings.image_embedding if hasattr(embeddings, 'image_embedding') and image_input else None
        text_embedding = embeddings.text_embedding if hasattr(embeddings, 'text_embedding') and contextual_text else None
        
        logger.info(f"Successfully retrieved embeddings. Image embedding present: {image_embedding is not None}. Text embedding present: {text_embedding is not None}")
        return image_embedding, text_embedding

    except Exception as e:
        logger.error(f"Failed to get multimodal embeddings: {e}")
        return None, None

if __name__ == '__main__':
    # Simple test (requires a local image 'sample_image.jpg' or a GCS URI)
    # Create a dummy sample_image.jpg for testing if you don't have one.
    # For example, using Pillow:
    # from PIL import Image as PILImage
    # img = PILImage.new('RGB', (60, 30), color = 'red')
    # img.save('sample_image.jpg')

    logger.info("Running gcp_clients.py self-test...")

    # Test 1: Text only
    print("\n--- Test 1: Text Embedding ---")
    _, text_emb = get_multimodal_embeddings(contextual_text="A delicious red apple on a wooden table.")
    if text_emb:
        print(f"Text embedding dimension: {len(text_emb)}")
        # print(f"Text embedding snippet: {text_emb[:5]}...")
    else:
        print("Failed to get text embedding.")

    # Test 2: Image only (provide a path to a sample image)
    # Ensure 'sample_image.jpg' exists in the ai_product_expert_bot directory or provide a valid path
    sample_image_file = "sample_image.jpg" 
    # Remove the incorrect text file creation - will create proper image below


    print("\n--- Test 2: Image Embedding ---")
    if_image_exists = False
    try:
        # Create a proper test image using PIL
        from PIL import Image as PILImage
        try:
            # Create a simple test image
            img_pil = PILImage.new('RGB', (100, 100), color='blue')
            img_pil.save(sample_image_file, 'JPEG')
            if_image_exists = True
            logger.info(f"Created a test image: {sample_image_file}")
        except Exception as e_pil:
            logger.error(f"Could not create test image using Pillow: {e_pil}. Ensure Pillow is installed.")

    except ImportError:
        logger.warning("Pillow not installed. Cannot create test image. Skipping image test.")


    if if_image_exists:
        img_emb, _ = get_multimodal_embeddings(image_path=sample_image_file)
        if img_emb:
            print(f"Image embedding dimension: {len(img_emb)}")
            # print(f"Image embedding snippet: {img_emb[:5]}...")
        else:
            print(f"Failed to get image embedding for {sample_image_file}.")
    else:
        print(f"Skipping image embedding test as {sample_image_file} could not be prepared/found.")


    # Test 3: Image and Text
    print("\n--- Test 3: Image and Text Embedding ---")
    if if_image_exists:
        img_emb_combo, text_emb_combo = get_multimodal_embeddings(
            image_path=sample_image_file,
            contextual_text="A vibrant blue square."
        )
        if img_emb_combo:
            print(f"Combo Image embedding dimension: {len(img_emb_combo)}")
        else:
            print(f"Failed to get combo image embedding for {sample_image_file}.")
        if text_emb_combo:
            print(f"Combo Text embedding dimension: {len(text_emb_combo)}")
        else:
            print("Failed to get combo text embedding.")
    else:
        print(f"Skipping image+text embedding test as {sample_image_file} could not be prepared/found.")

    # Clean up dummy image
    import os
    if if_image_exists and os.path.exists(sample_image_file):
        try:
            os.remove(sample_image_file)
            logger.info(f"Removed dummy image: {sample_image_file}")
        except OSError as e:
            logger.error(f"Error removing dummy image {sample_image_file}: {e}")

    # TODO: Add client for Gemini 2.5 Flash
    # TODO: Add client for text-embedding-005