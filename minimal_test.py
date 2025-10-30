#!/usr/bin/env python3
"""
Minimal test to isolate the segmentation fault issue
"""

import os
import sys

# Set environment variables BEFORE any imports
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

print("🧪 Step 1: Testing basic imports...")

try:
    import torch
    print("✅ torch imported")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print("✅ torch configured")
except Exception as e:
    print(f"❌ torch failed: {e}")
    sys.exit(1)

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("✅ HuggingFaceEmbeddings imported")
except Exception as e:
    print(f"❌ HuggingFaceEmbeddings failed: {e}")
    sys.exit(1)

try:
    from langchain_chroma import Chroma
    print("✅ Chroma imported")
except Exception as e:
    print(f"❌ Chroma failed: {e}")
    sys.exit(1)

print("\n🧪 Step 2: Testing embedding creation...")

try:
    # Create embeddings with minimal settings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'batch_size': 1}
    )
    print("✅ Embeddings created")
    
    # Test encoding
    result = embeddings.embed_query("test")
    print(f"✅ Embedding test successful, dimension: {len(result)}")
    
except Exception as e:
    print(f"❌ Embedding creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🧪 Step 3: Testing streamlit import...")

try:
    import streamlit as st
    print("✅ Streamlit imported successfully")
except Exception as e:
    print(f"❌ Streamlit import failed: {e}")
    sys.exit(1)

print("\n🎉 All tests passed! The issue might be in the actual app code.")