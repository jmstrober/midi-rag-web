#!/usr/bin/env python3
"""
Test the RAG engine directly
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

try:
    from rag_engine import RAGEngine
    
    print("🧪 Testing RAG Engine initialization...")
    
    rag = RAGEngine()
    print("✅ RAG Engine initialized")
    
    print(f"Vector store fallback: {rag.vector_store.use_fallback}")
    print(f"LLM client available: {hasattr(rag, 'client') and rag.client is not None}")
    
    # Test search
    print("\n🔍 Testing comprehensive search...")
    test_query = "menopause hormone replacement therapy"
    
    try:
        # Test the internal _comprehensive_search method
        search_results = rag._comprehensive_search(test_query, k=5)
        print(f"Comprehensive search returned: {len(search_results)} results")
        
        for i, (doc, score) in enumerate(search_results[:3]):
            print(f"Result {i+1}: Score={score:.3f}, Content={doc.page_content[:100]}...")
    
    except Exception as e:
        print(f"❌ Comprehensive search failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test full query
    print("\n🔍 Testing full RAG query...")
    try:
        result = rag.query("What are the risks of hormone replacement therapy for menopause?")
        print(f"Query result type: {type(result)}")
        print(f"Answer length: {len(result.get('answer', ''))}")
        print(f"Sources count: {len(result.get('sources', []))}")
        print(f"Answer preview: {result.get('answer', '')[:200]}...")
        
    except Exception as e:
        print(f"❌ Full query failed: {e}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"❌ RAG Engine initialization failed: {e}")
    import traceback
    traceback.print_exc()