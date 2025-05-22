#!/usr/bin/env python3
"""
Streamlit Web Interface for AI Product Expert Bot
A user-friendly web interface for searching products using natural language queries.
"""

import streamlit as st
import time
import os
import json
import pandas as pd
from typing import Dict, List, Any
from PIL import Image
import tempfile

# Import our search modules
try:
    import search
    import vector_store
    import config
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.error("Please make sure you're running this from the correct directory with all dependencies installed.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="AI Product Expert Bot",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.search-container {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 2rem;
}

.result-card {
    border: 1px solid #e9ecef;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    background: white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.metric-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
}

.sidebar-info {
    background: #e8f4f8;
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    if 'search_performed' not in st.session_state:
        st.session_state.search_performed = False

def get_database_stats():
    """Get statistics about the product database."""
    try:
        collection_info = vector_store.get_collection_info()
        if collection_info:
            return {
                'total_products': collection_info.get('points_count', 0),
                'status': collection_info.get('status', 'unknown'),
                'collection_name': collection_info.get('name', 'unknown')
            }
    except Exception as e:
        st.warning(f"Could not retrieve database stats: {e}")
    return {'total_products': 'Unknown', 'status': 'Unknown', 'collection_name': 'Unknown'}

def format_price(price_str: str, price_val: float = None) -> str:
    """Format price for display."""
    if price_val:
        return f"{price_str} ({price_val:.2f} DKK)"
    return price_str or "Price not available"

def display_product_card(result: Dict[str, Any], index: int):
    """Display a single product result as a card."""
    payload = result.get('payload', {})
    score = result.get('score', 0.0)
    
    with st.container():
        # Create columns for layout
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            # Product image
            image_url = payload.get('image_link')
            if image_url:
                try:
                    st.image(image_url, width=150, caption="Product Image")
                except:
                    st.write("🖼️ Image unavailable")
            else:
                st.write("🖼️ No image")
        
        with col2:
            # Product details
            st.markdown(f"### {payload.get('title', 'Unknown Product')}")
            
            # Create metrics row
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("Brand", payload.get('brand', 'Unknown'))
            
            with metric_col2:
                price_str = format_price(payload.get('price_str', ''), payload.get('price_val'))
                st.metric("Price", price_str)
            
            with metric_col3:
                availability = payload.get('availability', 'unknown')
                color = "🟢" if availability == "in_stock" else "🔴" if availability == "out_of_stock" else "🟡"
                st.metric("Availability", f"{color} {availability.replace('_', ' ').title()}")
            
            # Additional info
            if payload.get('product_type_specific'):
                st.write(f"**Category:** {payload['product_type_specific']}")
            
            if payload.get('description'):
                with st.expander("📄 Product Description"):
                    st.write(payload['description'])
        
        with col3:
            # Score and actions
            st.metric("Relevance Score", f"{score:.3f}")
            
            # Link to product
            if payload.get('link'):
                st.link_button("🔗 View Product", payload['link'])
            
            # Product ID for reference
            if payload.get('product_id'):
                st.caption(f"ID: {payload['product_id']}")

def perform_search(query: str, limit: int, image_path: str = None) -> List[Dict[str, Any]]:
    """Perform the actual search with error handling."""
    try:
        # Validate configuration
        config.validate_config()
        
        # Perform search
        results = search.search_with_llm_parsed_query(
            natural_language_query=query,
            image_path_for_query=image_path,
            limit=limit
        )
        
        return results
    
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        st.error("Please check your configuration and try again.")
        return []

def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛍️ AI Product Expert Bot</h1>
        <p>Find products using natural language in Danish or English</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with info and settings
    with st.sidebar:
        st.markdown("## ⚙️ Search Settings")
        
        # Database stats
        st.markdown("### 📊 Database Info")
        db_stats = get_database_stats()
        
        st.markdown(f"""
        <div class="sidebar-info">
        <strong>Collection:</strong> {db_stats['collection_name']}<br>
        <strong>Total Products:</strong> {db_stats['total_products']}<br>
        <strong>Status:</strong> {db_stats['status']}
        </div>
        """, unsafe_allow_html=True)
        
        # Search parameters
        st.markdown("### 🔧 Parameters")
        max_results = st.slider("Maximum Results", min_value=1, max_value=20, value=5)
        
        # Advanced options
        with st.expander("🔍 Advanced Options"):
            show_debug = st.checkbox("Show Debug Information")
            show_raw_results = st.checkbox("Show Raw Search Results")
        
        # Search history
        if st.session_state.search_history:
            st.markdown("### 📝 Recent Searches")
            for i, hist_query in enumerate(st.session_state.search_history[-5:]):
                if st.button(f"🔄 {hist_query}", key=f"hist_{i}"):
                    st.session_state.query_input = hist_query
                    st.experimental_rerun()
    
    # Main search interface
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    
    # Search mode selection
    search_mode = st.radio(
        "Search Mode",
        ["🔤 Text Only", "📸 Image Only", "🔤📸 Text + Image"],
        horizontal=True,
        help="Choose how you want to search: text only, image only, or combine both"
    )
    
    # Search input based on mode
    query = ""
    if search_mode in ["🔤 Text Only", "🔤📸 Text + Image"]:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "What are you looking for?",
                placeholder="Examples: 'stegepande under 500 kroner', 'Tefal kettle', 'coffee machine'",
                key="query_input",
                help="Try searching in Danish or English! Examples: 'kaffemaskine', 'frying pan under 400 DKK'"
            )
        
        with col2:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    else:
        # Image-only mode
        search_button = st.button("🔍 Search with Image", type="primary", use_container_width=True)
    
    # Image upload
    uploaded_image = None
    if search_mode in ["📸 Image Only", "🔤📸 Text + Image"]:
        uploaded_image = st.file_uploader(
            "📸 Upload an image" + (" (required for image-only search)" if search_mode == "📸 Image Only" else " (optional)"),
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Upload an image to search for visually similar products"
        )
        
        # Show preview of uploaded image
        if uploaded_image:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                image = Image.open(uploaded_image)
                st.image(image, caption="Search Image Preview", width=300)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Example queries
    st.markdown("### 💡 Try These Example Searches:")
    
    if search_mode in ["🔤 Text Only", "🔤📸 Text + Image"]:
        example_col1, example_col2, example_col3 = st.columns(3)
        
        with example_col1:
            if st.button("🍳 Stegepande under 500 kr"):
                st.session_state.query_input = "stegepande under 500 kroner"
                st.experimental_rerun()
        
        with example_col2:
            if st.button("🫖 Tefal kettle"):
                st.session_state.query_input = "Tefal kedel"
                st.experimental_rerun()
        
        with example_col3:
            if st.button("🦷 Ordo toothbrush"):
                st.session_state.query_input = "Ordo tandbørste"
                st.experimental_rerun()
    
    elif search_mode == "📸 Image Only":
        st.info("💡 **Image Search Tips:**")
        st.markdown("""
        - Upload clear, well-lit product images
        - Center the product in the frame
        - Avoid cluttered backgrounds
        - The system will find visually similar products
        - Works best with kitchen utensils, appliances, and tools
        """)
        
        # Show some example images from the database if available
        st.markdown("**Sample products you can search for with images:**")
        example_col1, example_col2, example_col3 = st.columns(3)
        
        with example_col1:
            st.markdown("🍳 **Frying Pans**")
            st.caption("Upload a photo of any frying pan")
        
        with example_col2:
            st.markdown("🫖 **Kettles & Teapots**")
            st.caption("Upload photos of kettles or teapots")
        
        with example_col3:
            st.markdown("🦷 **Personal Care**")
            st.caption("Upload photos of toothbrushes, etc.")
    
    # Perform search when button clicked
    if search_button:
        # Validate input based on search mode
        can_search = False
        search_description = ""
        
        if search_mode == "🔤 Text Only":
            can_search = query and query.strip()
            search_description = f"text query '{query}'"
            if not can_search:
                st.warning("⚠️ Please enter a search query")
        
        elif search_mode == "📸 Image Only":
            can_search = uploaded_image is not None
            search_description = "image search"
            if not can_search:
                st.warning("⚠️ Please upload an image for image-only search")
        
        elif search_mode == "🔤📸 Text + Image":
            can_search = (query and query.strip()) or uploaded_image is not None
            search_description = f"combined search"
            if query and query.strip() and uploaded_image:
                search_description = f"text '{query}' + image"
            elif query and query.strip():
                search_description = f"text '{query}'"
            elif uploaded_image:
                search_description = "image only"
            
            if not can_search:
                st.warning("⚠️ Please provide either text, image, or both for combined search")
        
        if can_search:
            # Save uploaded image temporarily if provided
            image_path = None
            if uploaded_image:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_image.type.split('/')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_image.read())
                    image_path = tmp_file.name
            
            try:
                # Show search progress
                with st.spinner(f"🔍 Searching with {search_description}..."):
                    start_time = time.time()
                    
                    # For image-only search, use empty query
                    search_query = query if search_mode != "📸 Image Only" else ""
                    
                    # Perform search
                    results = perform_search(search_query, max_results, image_path)
                    
                    search_time = time.time() - start_time
                
                # Clean up temporary image
                if image_path and os.path.exists(image_path):
                    os.remove(image_path)
                
                # Store results and update history
                st.session_state.search_results = results
                st.session_state.search_performed = True
                
                # Only add text queries to history
                if query and query.strip() and query not in st.session_state.search_history:
                    st.session_state.search_history.append(query)
                
                # Display search summary
                st.markdown("---")
                
                if results:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🎯 Results Found", len(results))
                    with col2:
                        st.metric("⏱️ Search Time", f"{search_time:.2f}s")
                    with col3:
                        avg_score = sum(r.get('score', 0) for r in results) / len(results)
                        st.metric("📊 Avg. Relevance", f"{avg_score:.3f}")
                    with col4:
                        mode_icon = "🔤" if search_mode == "🔤 Text Only" else "📸" if search_mode == "📸 Image Only" else "🔤📸"
                        st.metric("🔍 Search Mode", mode_icon)
                    
                    st.success(f"Found {len(results)} products using {search_description}")
                else:
                    st.warning("No products found. Try different search terms, upload a clearer image, or check the examples above.")
                
                # Debug information
                if show_debug and results:
                    with st.expander("🐛 Debug Information"):
                        st.write("**Search Configuration:**")
                        st.write(f"- Search mode: {search_mode}")
                        st.write(f"- Text query: {query if query else 'None'}")
                        st.write(f"- Image provided: {'Yes' if uploaded_image else 'No'}")
                        st.write(f"- Limit: {max_results}")
                        st.write(f"- Search time: {search_time:.2f} seconds")
                
            except Exception as e:
                st.error(f"❌ Search failed: {str(e)}")
                if show_debug:
                    st.exception(e)
    
    # Display results
    if st.session_state.search_performed and st.session_state.search_results:
        st.markdown("---")
        st.markdown("## 🛍️ Search Results")
        
        # Display each result
        for i, result in enumerate(st.session_state.search_results):
            with st.container():
                display_product_card(result, i + 1)
                if i < len(st.session_state.search_results) - 1:
                    st.markdown("---")
        
        # Raw results for debugging
        if show_raw_results:
            with st.expander("🔬 Raw Search Results (JSON)"):
                st.json(st.session_state.search_results)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🤖 Powered by AI Product Expert Bot | Built with Streamlit</p>
        <p>Supports Danish and English queries | Hybrid semantic + keyword search</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()