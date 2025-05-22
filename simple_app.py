#!/usr/bin/env python3
"""
Simple test Streamlit app to verify functionality
"""

import streamlit as st
import time

# Test basic Streamlit functionality
st.title("🛍️ AI Product Expert Bot - Test Interface")
st.write("This is a simple test to verify Streamlit is working properly.")

# Test search functionality
query = st.text_input("Enter a test query:", placeholder="stegepande under 500 kroner")

if st.button("Test Search"):
    if query:
        with st.spinner("Testing search functionality..."):
            time.sleep(2)  # Simulate processing
            
            # Test imports
            try:
                import search
                st.success("✅ Search module imported successfully")
                
                import vector_store
                st.success("✅ Vector store module imported successfully")
                
                import config
                config.validate_config()
                st.success("✅ Configuration validated successfully")
                
                st.info(f"🔍 Ready to search for: '{query}'")
                st.write("**Note:** This is just a test. The full interface has more features!")
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        st.warning("Please enter a search query")

# System info
st.markdown("---")
st.markdown("### System Status")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Streamlit", "✅ Running")

with col2:
    try:
        import search
        st.metric("Search Module", "✅ Loaded")
    except:
        st.metric("Search Module", "❌ Error")

with col3:
    try:
        import config
        config.validate_config()
        st.metric("Configuration", "✅ Valid")
    except:
        st.metric("Configuration", "❌ Error")

st.markdown("---")
st.info("💡 If this test works, the full app should work too!")
st.write("🔗 Try the full interface by running: `streamlit run streamlit_app.py`")