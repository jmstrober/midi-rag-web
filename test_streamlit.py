#!/usr/bin/env python3
"""
Test Streamlit without RAG engines
"""

import os
# Set environment variables early
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Configure PyTorch
try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.float32)
except ImportError:
    pass

import streamlit as st

# Configure page
st.set_page_config(
    page_title="MIDI Test App",
    page_icon="🏥",
    layout="wide"
)

st.title("🧪 MIDI RAG Test - No Engines")
st.write("This is a test version without RAG engine initialization.")

if st.button("Test Button"):
    st.success("✅ Streamlit is working!")

st.sidebar.title("Test Sidebar")
st.sidebar.write("If you can see this, Streamlit is running properly.")