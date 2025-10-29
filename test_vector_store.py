#!/usr/bin/env python3
"""
Quick test script to verify VectorStoreManager is working properly
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.VectorStoreManager import VectorStoreManager

def test_vector_store():
    print("🧪 Testing VectorStoreManager...")
    
    # Initialize the vector store
    vm = VectorStoreManager("./data/chroma_db")
    
    # Check if it's using fallback or embeddings
    print(f"📊 Using fallback: {vm.use_fallback}")
    
    if not vm.use_fallback:
        print("✅ Using embeddings!")
        if vm.collection:
            count = vm.collection.count()
            print(f"📚 Collection has {count} documents")
        else:
            print("❌ Collection is None")
    else:
        print("⚠️ Using fallback mode")
    
    # Test a search
    print("\n🔍 Testing search with query: 'breast cancer HRT'")
    results = vm.search_with_scores("breast cancer HRT", k=3)
    
    print(f"📋 Found {len(results)} results:")
    for i, (doc, score) in enumerate(results[:2]):
        print(f"  {i+1}. Score: {score:.3f}")
        print(f"     Content preview: {doc.page_content[:100]}...")
        print(f"     Metadata: {doc.metadata}")
        print()

if __name__ == "__main__":
    test_vector_store()