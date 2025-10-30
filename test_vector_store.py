#!/usr/bin/env python3
"""
Test the VectorStoreManager directly to see what's failing
"""

import sys
from pathlib import Path
import os

# Add src to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "src"))

# Set environment variables early
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

from src.utils.VectorStoreManager import VectorStoreManager

def test_vector_store():
    print("🧪 Testing VectorStoreManager initialization...")
    
    try:
        # Initialize VectorStoreManager
        vector_store = VectorStoreManager("./data/chroma_db")
        
        print(f"✅ VectorStoreManager initialized")
        print(f"   use_fallback: {vector_store.use_fallback}")
        print(f"   embeddings: {vector_store.embeddings is not None}")
        print(f"   vectorstore: {vector_store.vectorstore is not None}")
        
        # Test search
        print(f"\n🔍 Testing search...")
        test_query = "family history breast cancer HRT hormone replacement therapy"
        
        try:
            results = vector_store.search_with_scores(
                query=test_query,
                k=5
            )
            
            print(f"✅ Search returned {len(results)} results")
            
            for i, (doc, score) in enumerate(results[:3]):
                print(f"\nResult {i+1} (score: {score:.3f}):")
                print(f"  Content: {doc.page_content[:200]}...")
                print(f"  Metadata: {doc.metadata}")
                
        except Exception as e:
            print(f"❌ Search failed: {e}")
            import traceback
            traceback.print_exc()
            
        # Test collection stats
        try:
            if hasattr(vector_store, 'get_collection_stats'):
                stats = vector_store.get_collection_stats()
                print(f"\n📊 Collection stats: {stats}")
            elif hasattr(vector_store, 'fallback_store') and hasattr(vector_store.fallback_store, 'get_collection_stats'):
                stats = vector_store.fallback_store.get_collection_stats()
                print(f"\n📊 Fallback stats: {stats}")
        except Exception as e:
            print(f"⚠️ Could not get stats: {e}")
            
    except Exception as e:
        print(f"❌ VectorStoreManager initialization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vector_store()