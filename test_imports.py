#!/usr/bin/env python3
"""
Test script to verify imports work correctly
"""

import os
# Set environment variables early
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

def test_imports():
    print("🧪 Testing imports...")
    
    try:
        print("1. Testing torch import...")
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.set_default_device('cpu')
        print("   ✅ torch imported successfully")
    except Exception as e:
        print(f"   ❌ torch import failed: {e}")
        return False
    
    try:
        print("2. Testing langchain_huggingface import...")
        from langchain_huggingface import HuggingFaceEmbeddings
        print("   ✅ langchain_huggingface imported successfully")
    except Exception as e:
        print(f"   ❌ langchain_huggingface import failed: {e}")
        try:
            print("   🔄 Trying fallback import...")
            from langchain_community.embeddings import HuggingFaceEmbeddings
            print("   ✅ langchain_community.embeddings fallback successful")
        except Exception as e2:
            print(f"   ❌ Fallback also failed: {e2}")
            return False
    
    try:
        print("3. Testing langchain_chroma import...")
        from langchain_chroma import Chroma
        print("   ✅ langchain_chroma imported successfully")
    except Exception as e:
        print(f"   ❌ langchain_chroma import failed: {e}")
        try:
            print("   🔄 Trying fallback import...")
            from langchain_community.vectorstores import Chroma
            print("   ✅ langchain_community.vectorstores fallback successful")
        except Exception as e2:
            print(f"   ❌ Fallback also failed: {e2}")
            return False
    
    try:
        print("4. Testing sentence_transformers import...")
        from sentence_transformers import SentenceTransformer
        print("   ✅ sentence_transformers imported successfully")
    except Exception as e:
        print(f"   ❌ sentence_transformers import failed: {e}")
        return False
    
    try:
        print("5. Testing basic embedding creation...")
        model_kwargs = {'device': 'cpu', 'trust_remote_code': False}
        encode_kwargs = {'normalize_embeddings': True, 'batch_size': 1}
        
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        # Test with a simple query
        test_embedding = embeddings.embed_query("test query")
        print(f"   ✅ Embedding created successfully, dimension: {len(test_embedding)}")
    except Exception as e:
        print(f"   ❌ Embedding creation failed: {e}")
        return False
    
    print("\n🎉 All imports successful!")
    return True

if __name__ == "__main__":
    success = test_imports()
    if not success:
        print("\n❌ Import test failed. Please check your environment.")
        exit(1)
    else:
        print("\n✅ Import test passed. You can now run the Streamlit app.")