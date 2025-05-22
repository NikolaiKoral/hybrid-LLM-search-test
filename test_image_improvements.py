#!/usr/bin/env python3
"""
Test the improved image search functionality
"""

import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

import search
import config

def test_improved_image_search():
    """Test the improved image search"""
    print("=== Testing Improved Image Search ===")
    
    # Validate config first
    try:
        config.validate_config()
        print("✓ Configuration validated")
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return
    
    # Test with one image
    sample_image = "images/5714988014434.jpg"
    if not Path(sample_image).exists():
        print(f"✗ Sample image not found: {sample_image}")
        return
    
    print(f"\n--- Testing Image-Only Search with {Path(sample_image).name} ---")
    
    try:
        # Test image-only search with improved weights
        print("Testing improved image-only search...")
        results = search.search_with_llm_parsed_query(
            natural_language_query="",
            image_path_for_query=sample_image,
            limit=5
        )
        
        print(f"Found {len(results)} results")
        for j, result in enumerate(results):
            score = result.get('score', 0)
            payload = result.get('payload', {})
            title = payload.get('title', 'Unknown')
            product_type = payload.get('product_type_specific', 'Unknown')
            brand = payload.get('brand', 'Unknown')
            print(f"  {j+1}. {title}")
            print(f"      Brand: {brand} | Type: {product_type} | Score: {score:.4f}")
        
        # Test text + image search 
        print(f"\n--- Testing Text + Image Search ---")
        results = search.search_with_llm_parsed_query(
            natural_language_query="stegepande under 500 kr",
            image_path_for_query=sample_image,
            limit=5
        )
        
        print(f"Found {len(results)} results")
        for j, result in enumerate(results):
            score = result.get('score', 0)
            payload = result.get('payload', {})
            title = payload.get('title', 'Unknown')
            product_type = payload.get('product_type_specific', 'Unknown')
            brand = payload.get('brand', 'Unknown')
            price = payload.get('price_val', 'Unknown')
            print(f"  {j+1}. {title}")
            print(f"      Brand: {brand} | Type: {product_type} | Price: {price} | Score: {score:.4f}")
                
    except Exception as e:
        print(f"✗ Error during search: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_improved_image_search()