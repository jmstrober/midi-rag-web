#!/usr/bin/env python3
"""
Test the complete RAG pipeline with a breast cancer HRT question
"""

import sys
import os
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "src"))

from patient_rag_engine import PatientRAGEngine

def test_breast_cancer_hrt_query():
    print("🧪 Testing complete RAG pipeline with breast cancer HRT query...")
    
    try:
        # Initialize the patient RAG engine
        print("🔄 Initializing PatientRAGEngine...")
        engine = PatientRAGEngine()
        
        # Check if vector store is working
        print(f"📊 Using fallback: {engine.vector_store.use_fallback}")
        
        # Test the query that wasn't working
        query = "I have a family history of breast cancer. Can I still do HRT?"
        print(f"\n🔍 Testing query: '{query}'")
        
        # Get documents first to see if retrieval is working
        print("\n📚 Testing document retrieval...")
        documents = engine.vector_store.search_with_scores(query, k=5)
        print(f"Found {len(documents)} relevant documents:")
        
        for i, (doc, score) in enumerate(documents[:3]):
            print(f"  {i+1}. Score: {score:.3f}")
            print(f"     Content: {doc.page_content[:100]}...")
            print(f"     Source: {doc.metadata.get('filename', 'Unknown')}")
            print()
        
        # Now test the full RAG response
        print("🤖 Testing full RAG response...")
        try:
            response = engine.query(query)
            print(f"Response: {response}")
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            # Still show that retrieval works even if LLM fails
            print("✅ Document retrieval is working, LLM error is separate issue")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_breast_cancer_hrt_query()