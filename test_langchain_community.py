#!/usr/bin/env python3
"""
Test script to verify langchain_community works correctly
"""
import warnings
warnings.filterwarnings('ignore')
import sys
import os

# Set offline mode
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

# Add src to path
sys.path.append('src')

def test_embeddings():
    print("🔍 Testing langchain_community embeddings...")
    
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from sentence_transformers import SentenceTransformer
        
        print("✅ Imports successful")
        
        # Test embeddings initialization
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Test encoding
        test_text = "This is a test query"
        test_embedding = embeddings.embed_query(test_text)
        
        print(f"✅ Embedding generated, dimension: {len(test_embedding)}")
        print("✅ langchain_community embeddings working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_vector_store_manager():
    print("\n🔍 Testing VectorStoreManager with langchain_community...")
    
    try:
        from utils.VectorStoreManager import VectorStoreManager
        
        # Initialize (but don't load full database to save time)
        manager = VectorStoreManager(persist_directory="data/chroma_db")
        print("✅ VectorStoreManager initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing langchain_community compatibility...")
    
    success = True
    success &= test_embeddings()
    success &= test_vector_store_manager()
    
    if success:
        print("\n🎉 All tests passed! Ready for Streamlit Cloud deployment.")
    else:
        print("\n❌ Some tests failed. Check the errors above.")