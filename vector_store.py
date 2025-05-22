"""
Qdrant vector database service for the AI Product Expert Bot.
Handles dense and sparse vectors for hybrid search.
"""

import logging
import time
import json
import uuid
from typing import Dict, List, Optional, Union, Any

import qdrant_client
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

# Import config
try:
    import config
except ModuleNotFoundError:
    print("ERROR: config.py not found. Please ensure it's in the same directory or PYTHONPATH is set.")
    # Fallback for critical configs if needed
    config = type('obj', (object,), {
        'QDRANT_URL': 'http://localhost:6333',
        'QDRANT_API_KEY': None,
        'QDRANT_COLLECTION_NAME': 'ai_product_expert_collection',
        'DEFAULT_DENSE_VECTOR_NAME': 'holistic_dense',
        'SPARSE_VECTOR_NAME': 'minicoil_sparse',
        'HOLISTIC_DENSE_VECTOR_DIM': 3667 # Example, ensure this matches your actual dimension
    })

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_qdrant_client = None

def get_qdrant_client() -> qdrant_client.QdrantClient:
    """Returns a singleton instance of the Qdrant client with validation."""
    global _qdrant_client
    if _qdrant_client is None:
        try:
            # Validate configuration
            if not config.QDRANT_URL:
                raise ValueError("QDRANT_URL is not configured")
            
            logger.info(f"Initializing Qdrant client with URL: {config.QDRANT_URL}")
            _qdrant_client = qdrant_client.QdrantClient(
                url=config.QDRANT_URL,
                api_key=config.QDRANT_API_KEY,
                timeout=30.0  # Increased timeout for reliability
            )
            
            # Test connection
            try:
                _ = _qdrant_client.get_collections()
                logger.info("Qdrant client initialized and connection verified.")
            except Exception as e:
                logger.warning(f"Qdrant connection test failed: {e}")
                # Continue anyway as collections might not exist yet
                
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            _qdrant_client = None
            raise
    return _qdrant_client

def ensure_collection_exists(
    collection_name: str = config.QDRANT_COLLECTION_NAME,
    dense_vector_name: str = config.DEFAULT_DENSE_VECTOR_NAME,
    dense_vector_size: int = config.HOLISTIC_DENSE_VECTOR_DIM,
    dense_distance: models.Distance = models.Distance.COSINE,
    sparse_vector_name: Optional[str] = config.SPARSE_VECTOR_NAME
) -> bool:
    """
    Ensures that the specified collection exists in Qdrant with named dense and sparse vector configurations.
    If it doesn't exist, creates it.
    """
    client = get_qdrant_client()
    try:
        collections_response = client.get_collections()
        collection_names = [c.name for c in collections_response.collections]
        
        # Prepare dense vectors configuration
        dense_vectors_payload: Dict[str, models.VectorParams] = {
            dense_vector_name: models.VectorParams(
                size=dense_vector_size,
                distance=dense_distance
            )
        }
        
        # Prepare sparse vectors configuration
        sparse_vectors_payload: Optional[Dict[str, models.SparseVectorParams]] = None
        if sparse_vector_name:
            sparse_vectors_payload = {
                sparse_vector_name: models.SparseVectorParams()
                # Qdrant applies IDF scoring for sparse vectors by default.
                # Specific index parameters (like modifier) are for payload indexing on sparse indices,
                # not part of the direct SparseVectorParams for collection creation.
            }

        if collection_name in collection_names:
            logger.info(f"Collection '{collection_name}' already exists.")
            # TODO: Add robust verification of existing collection's dense and sparse vector configurations.
            # This might involve comparing client.get_collection(collection_name).config.params.vectors
            # and .config.params.sparse_vectors with the desired configs.
            # If mismatch, could raise error or attempt recreation (carefully).
            # For now, data_ingestion.py's logic handles some recreation scenarios.
            return True
        
        log_message_parts = [f"dense vector '{dense_vector_name}' (size {dense_vector_size}, dist {dense_distance})"]
        if sparse_vectors_payload:
            log_message_parts.append(f"sparse vector '{sparse_vector_name}'")
        logger.info(f"Creating collection '{collection_name}' with " + " and ".join(log_message_parts) + "...")
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=dense_vectors_payload,      # Corrected: use 'vectors_config' for dense
            sparse_vectors_config=sparse_vectors_payload # This was already correct
        )
        logger.info(f"Collection '{collection_name}' created successfully.")
        
        # Create payload indexes for brand and price_val
        try:
            logger.info(f"Creating payload index for 'brand' in collection '{collection_name}'...")
            client.create_payload_index(
                collection_name=collection_name,
                field_name="brand",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            logger.info(f"Payload index for 'brand' created successfully.")

            logger.info(f"Creating payload index for 'price_val' in collection '{collection_name}'...")
            client.create_payload_index(
                collection_name=collection_name,
                field_name="price_val",
                field_schema=models.PayloadSchemaType.FLOAT
            )
            logger.info(f"Payload index for 'price_val' created successfully.")
            
        except Exception as index_e:
            logger.error(f"Failed to create payload indexes for collection '{collection_name}': {index_e}", exc_info=True)
            # Depending on strictness, we might want to return False or raise here
            # For now, log error and continue (collection exists, but filtering might fail)

        return True
    except Exception as e:
        logger.error(f"Failed to ensure collection '{collection_name}' exists: {e}", exc_info=True)
        return False

def upsert_points(
    points_data: List[Dict[str, Any]], # Each dict: {'id': str, 'payload': dict, 'dense_vector': List[float], 'sparse_vector': {'indices': List[int], 'values': List[float]}}
    collection_name: str = config.QDRANT_COLLECTION_NAME,
    dense_vector_name: str = config.DEFAULT_DENSE_VECTOR_NAME,
    sparse_vector_name: Optional[str] = config.SPARSE_VECTOR_NAME,
    batch_size: int = 100
) -> bool:
    """
    Inserts or updates points with named dense and (optional) sparse vectors using batching.
    """
    client = get_qdrant_client()
    
    if not points_data:
        logger.info("No points provided for upsert.")
        return True
    
    # Process points in batches for better performance and memory management
    total_points = len(points_data)
    total_processed = 0
    total_skipped = 0
    
    for batch_start in range(0, total_points, batch_size):
        batch_end = min(batch_start + batch_size, total_points)
        batch_data = points_data[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start//batch_size + 1}/{(total_points-1)//batch_size + 1} ({len(batch_data)} points)")
        
        qdrant_points: List[models.PointStruct] = []
        
        for point_data in batch_data:
            # Input validation
            if not isinstance(point_data, dict) or 'id' not in point_data:
                logger.warning(f"Invalid point data structure, skipping")
                total_skipped += 1
                continue
                
            vector_map: Dict[str, Union[List[float], models.SparseVector]] = {}
            
            # Process dense vector with validation
            if 'dense_vector' in point_data and point_data['dense_vector'] is not None:
                dense_vec = point_data['dense_vector']
                if isinstance(dense_vec, list) and len(dense_vec) > 0:
                    vector_map[dense_vector_name] = dense_vec
                else:
                    logger.warning(f"Point {point_data['id']} has invalid dense vector format")
            
            # Process sparse vector with validation
            if sparse_vector_name and 'sparse_vector' in point_data and point_data['sparse_vector'] is not None:
                sparse_data = point_data['sparse_vector']
                if isinstance(sparse_data, dict) and "indices" in sparse_data and "values" in sparse_data:
                    try:
                        indices = sparse_data['indices']
                        values = sparse_data['values']
                        
                        # Validate sparse vector data
                        if isinstance(indices, list) and isinstance(values, list) and len(indices) == len(values):
                            vector_map[sparse_vector_name] = models.SparseVector(
                                indices=indices,
                                values=values
                            )
                        else:
                            logger.warning(f"Point {point_data['id']} has mismatched sparse vector dimensions")
                    except Exception as e:
                        logger.warning(f"Point {point_data['id']} sparse vector creation failed: {e}")
                else:
                    logger.warning(f"Point {point_data['id']} has invalid sparse_vector format")
            
            if not vector_map:
                logger.warning(f"Point {point_data['id']} has no valid vectors to upsert. Skipping point.")
                total_skipped += 1
                continue

            # Validate payload
            payload = point_data.get('payload', {})
            if not isinstance(payload, dict):
                logger.warning(f"Point {point_data['id']} has invalid payload format, using empty payload")
                payload = {}

            qdrant_points.append(models.PointStruct(
                id=point_data['id'],
                payload=payload,
                vector=vector_map
            ))
            
        if not qdrant_points:
            logger.info(f"No valid points in batch {batch_start//batch_size + 1}, skipping")
            continue

        # Upsert batch with retry logic
        batch_success = False
        max_retries = 3
        
        for retry in range(max_retries):
            try:
                client.upsert(collection_name=collection_name, points=qdrant_points, wait=True)
                batch_success = True
                total_processed += len(qdrant_points)
                logger.info(f"Successfully upserted batch {batch_start//batch_size + 1} ({len(qdrant_points)} points)")
                break
            except Exception as e:
                logger.warning(f"Batch upsert attempt {retry + 1} failed: {e}")
                if retry < max_retries - 1:
                    time.sleep(1 * (retry + 1))  # Exponential backoff
                else:
                    logger.error(f"Failed to upsert batch {batch_start//batch_size + 1} after {max_retries} attempts")
                    return False
        
        # Brief pause between batches to avoid overwhelming the server
        if batch_end < total_points:
            time.sleep(0.1)
    
    logger.info(f"Upsert completed: {total_processed} points processed, {total_skipped} skipped")
    return True

def search_points(
    dense_query_vector: Optional[List[float]] = None,
    sparse_query_vector_data: Optional[Dict[str, Any]] = None, # {"indices": List[int], "values": List[float]}
    limit: int = 10,
    collection_name: str = config.QDRANT_COLLECTION_NAME,
    dense_vector_name: str = config.DEFAULT_DENSE_VECTOR_NAME,
    sparse_vector_name: Optional[str] = config.SPARSE_VECTOR_NAME,
    filter_condition: Optional[models.Filter] = None,
    with_payload: bool = True,
    with_vectors: bool = False
) -> List[Dict[str, Any]]:
    """
    Performs a hybrid search using dense and/or sparse vectors.
    Uses separate searches for dense and sparse vectors, then combines results.
    """
    client = get_qdrant_client()
    
    if not dense_query_vector and not sparse_query_vector_data:
        logger.error("At least one of dense_query_vector or sparse_query_vector_data must be provided for search.")
        return []

    # Using two separate searches and manual fusion
    id_to_hit_map: Dict[Union[str, int], models.ScoredPoint] = {}
    id_to_scores_map: Dict[Union[str, int], Dict[str, float]] = {}

    # 1. Dense Search
    if dense_query_vector and dense_vector_name:
        logger.info(f"Performing dense search on '{dense_vector_name}'...")
        try:
            dense_hits = client.search(
                collection_name=collection_name,
                query_vector=models.NamedVector(name=dense_vector_name, vector=dense_query_vector),
                query_filter=filter_condition,
                limit=limit * 2,  # Fetch more to allow for better fusion
                with_payload=with_payload,
                with_vectors=with_vectors
            )
            for hit in dense_hits:
                id_to_hit_map[hit.id] = hit
                id_to_scores_map.setdefault(hit.id, {})['dense'] = hit.score
            logger.info(f"Dense search found {len(dense_hits)} hits.")
        except Exception as e:
            logger.error(f"Dense search failed: {e}", exc_info=True)

    # 2. Sparse Search
    if sparse_query_vector_data and sparse_vector_name:
        logger.info(f"Performing sparse search on '{sparse_vector_name}'...")
        try:
            # Create the sparse vector
            sparse_q_vec = models.SparseVector(
                indices=sparse_query_vector_data['indices'],
                values=sparse_query_vector_data['values']
            )
            
            # Create a NamedSparseVector for the query
            named_sparse_query_vector = models.NamedSparseVector(
                name=sparse_vector_name,
                vector=sparse_q_vec
            )
            
            sparse_hits = client.search(
                collection_name=collection_name,
                query_vector=named_sparse_query_vector,  # Pass the NamedSparseVector object
                query_filter=filter_condition,
                limit=limit * 2,
                with_payload=with_payload,
                with_vectors=with_vectors
            )
            
            for hit in sparse_hits:
                if hit.id not in id_to_hit_map:  # If new, store it
                    id_to_hit_map[hit.id] = hit
                id_to_scores_map.setdefault(hit.id, {})['sparse'] = hit.score
            logger.info(f"Sparse search found {len(sparse_hits)} hits.")
        except Exception as e:
            # Log the error but continue with dense results only
            logger.warning(f"Sparse search failed, continuing with dense results only: {str(e)}")
            logger.debug(f"Sparse search error details:", exc_info=True)

    # 3. Fusion (Simple RRF-like)
    # For now, let's use a weighted sum of normalized scores if available, or just combine.
    fused_scored_results = []
    for hit_id, scores_dict in id_to_scores_map.items():
        qdrant_hit = id_to_hit_map[hit_id]
        # Simple combined score for ranking - this is NOT RRF.
        # A proper RRF would require ranks from each list.
        # For now, let's use the score from the primary type of match or sum.
        # Max score is often a simple way to combine if scales are comparable.
        # Qdrant scores are distances/similarities, higher is better.
        final_score = max(scores_dict.get('dense', -float('inf')), scores_dict.get('sparse', -float('inf')))
        
        fused_scored_results.append({
            'id': qdrant_hit.id,
            'score': final_score,  # Using max score for simplicity
            'payload': qdrant_hit.payload if with_payload else None,
            'vector': qdrant_hit.vector if with_vectors else None
        })

    # Sort by the fused score
    fused_scored_results.sort(key=lambda x: x['score'], reverse=True)
    
    final_results = fused_scored_results[:limit]
    
    logger.info(f"Hybrid search post-fusion returned {len(final_results)} points.")
    return final_results



def delete_collection(collection_name: str = config.QDRANT_COLLECTION_NAME) -> bool:
    """Deletes the specified collection."""
    client = get_qdrant_client()
    try:
        logger.info(f"Deleting collection '{collection_name}'...")
        client.delete_collection(collection_name=collection_name)
        logger.info(f"Collection '{collection_name}' deleted successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to delete collection: {e}", exc_info=True)
        return False

def get_collection_info(collection_name: str = config.QDRANT_COLLECTION_NAME) -> Optional[Dict[str, Any]]:
    """Gets information about the specified collection."""
    client = get_qdrant_client()
    try:
        logger.info(f"Getting information for collection '{collection_name}'...")
        collection_info_model = client.get_collection(collection_name=collection_name)
        
        # Access vector configuration
        vectors_config = collection_info_model.config.params.vectors
        vector_size_info = "Multiple named vectors"
        distance_info = "Multiple named vectors"

        if isinstance(vectors_config, models.VectorParams): # Single unnamed vector
            vector_size_info = str(vectors_config.size)
            distance_info = str(vectors_config.distance)
        elif isinstance(vectors_config, dict): # Named vectors
            sizes = []
            distances = []
            for name, params in vectors_config.items():
                if isinstance(params, models.VectorParams):
                    sizes.append(f"{name}: {params.size}")
                    distances.append(f"{name}: {params.distance}")
                elif isinstance(params, models.SparseVectorParams):
                    sizes.append(f"{name}: Sparse")
                    distances.append(f"{name}: Sparse")
            vector_size_info = ", ".join(sizes)
            distance_info = ", ".join(distances)

        return {
            'name': collection_name,
            'vectors_count': collection_info_model.vectors_count if hasattr(collection_info_model, 'vectors_count') else collection_info_model.points_count, # points_count is often more relevant
            'points_count': collection_info_model.points_count,
            'status': str(collection_info_model.status),
            'vector_size_info': vector_size_info, # Detailed info
            'distance_info': distance_info,     # Detailed info
            'optimizer_status': str(collection_info_model.optimizer_status),
            'payload_schema': {k: str(v.data_type) for k, v in collection_info_model.payload_schema.items()} if collection_info_model.payload_schema else {}
        }
    except UnexpectedResponse as e:
        if "Not found" in str(e) or e.status_code == 404:
            logger.warning(f"Collection '{collection_name}' does not exist.")
            return None
        logger.error(f"Failed to get collection info (UnexpectedResponse): {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    # This test section needs significant updates for hybrid search
    logger.info("Testing Qdrant client (hybrid setup)...")
    
    try:
        client = get_qdrant_client()
        logger.info("Successfully connected to Qdrant.")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        exit(1)
    
    collection_name = config.QDRANT_COLLECTION_NAME
    
    # For testing, let's ensure collection is clean and correctly configured
    existing_info = get_collection_info(collection_name)
    if existing_info:
        logger.info(f"Collection '{collection_name}' exists. Deleting for fresh test setup.")
        delete_collection(collection_name)

    if not ensure_collection_exists(
        collection_name=collection_name,
        dense_vector_size=config.HOLISTIC_DENSE_VECTOR_DIM # Ensure this is correct
    ):
        logger.error(f"Failed to ensure collection '{collection_name}' exists for hybrid test.")
        exit(1)
    
    collection_info = get_collection_info(collection_name)
    if collection_info:
        logger.info(f"Collection info after ensure_collection_exists: {collection_info}")
    else:
        logger.error("Failed to get collection info after creation.")
        exit(1)

    # Test upserting points with dense and sparse vectors
    import uuid
    test_points_data = []
    # Example point
    test_points_data.append({
        'id': str(uuid.uuid4()),
        'dense_vector': [0.1] * config.HOLISTIC_DENSE_VECTOR_DIM, # Example dense vector
        'sparse_vector': {'indices': [10, 20, 30], 'values': [0.5, 0.3, 0.2]}, # Example sparse
        'payload': {'product_id': 'hybrid_test_01', 'name': 'Hybrid Test Product 1'}
    })
    test_points_data.append({ # Point with only dense
        'id': str(uuid.uuid4()),
        'dense_vector': [0.2] * config.HOLISTIC_DENSE_VECTOR_DIM,
        'sparse_vector': None, # Test missing sparse
        'payload': {'product_id': 'hybrid_test_02', 'name': 'Dense Only Product 2'}
    })
    test_points_data.append({ # Point with only sparse (less common but testable)
        'id': str(uuid.uuid4()),
        'dense_vector': None,
        'sparse_vector': {'indices': [15, 25], 'values': [0.8, 0.1]},
        'payload': {'product_id': 'hybrid_test_03', 'name': 'Sparse Only Product 3'}
    })


    if not upsert_points(test_points_data, collection_name):
        logger.error("Failed to upsert hybrid test points.")
        exit(1)
    
    logger.info(f"Upserted {len(test_points_data)} test points. Verifying count...")
    time.sleep(1) # Give Qdrant a moment
    collection_info_after_upsert = get_collection_info(collection_name)
    if collection_info_after_upsert:
        logger.info(f"Points count after upsert: {collection_info_after_upsert.get('points_count')}")
    else:
        logger.warning("Could not get collection info after upsert.")


    # Test searching (hybrid)
    dense_q_vec = [0.11] * config.HOLISTIC_DENSE_VECTOR_DIM # Slightly different from first point's dense
    sparse_q_data = {'indices': [10, 25, 40], 'values': [0.6, 0.4, 0.1]}

    logger.info("Testing hybrid search (dense + sparse):")
    search_results = search_points(
        dense_query_vector=dense_q_vec,
        sparse_query_vector_data=sparse_q_data,
        limit=5,
        collection_name=collection_name
    )
    if search_results: logger.info(f"Hybrid search results: {json.dumps(search_results, indent=2)}")
    else: logger.warning("No hybrid search results found.")

    logger.info("Testing dense-only search:")
    search_results_dense_only = search_points(
        dense_query_vector=dense_q_vec,
        limit=5,
        collection_name=collection_name
    )
    if search_results_dense_only: logger.info(f"Dense-only search results: {json.dumps(search_results_dense_only, indent=2)}")
    else: logger.warning("No dense-only search results found.")

    logger.info("Testing sparse-only search:")
    search_results_sparse_only = search_points(
        sparse_query_vector_data=sparse_q_data,
        limit=5,
        collection_name=collection_name
    )
    if search_results_sparse_only: logger.info(f"Sparse-only search results: {json.dumps(search_results_sparse_only, indent=2)}")
    else: logger.warning("No sparse-only search results found.")
    
    logger.info("Qdrant client hybrid tests completed.")
