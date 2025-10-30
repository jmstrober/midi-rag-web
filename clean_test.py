#!/usr/bin/env python3
"""
Clean Streamlit test - NO PyTorch configuration
"""

import os
# Only set these basic environment variables
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import streamlit as st

# Configure page
st.set_page_config(
    page_title="MIDI Test - Clean",
    page_icon="🏥",
    layout="wide"
)

st.title("🧪 MIDI RAG - Clean Test")
st.write("This version doesn't configure PyTorch threading.")

if st.button("Test Import"):
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        st.success("✅ langchain_huggingface imported successfully!")
    except Exception as e:
        st.error(f"❌ Import failed: {e}")

if st.button("Test Embeddings"):
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        
        # Create embeddings WITHOUT any torch configuration
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        result = embeddings.embed_query("test")
        st.success(f"✅ Embeddings created! Dimension: {len(result)}")
        
    except Exception as e:
        st.error(f"❌ Embeddings failed: {e}")

st.sidebar.title("Clean Test")
st.sidebar.write("No PyTorch threading configuration.")