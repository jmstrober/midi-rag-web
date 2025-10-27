#!/usr/bin/env python3
"""
Migration script to re-process existing documents with medical embeddings
"""

import sys
from pathlib import Path
import logging
import json
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.VectorStoreManager import VectorStoreManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def migrate_to_medical_embeddings():
    """Migrate existing ChromaDB to use medical embeddings."""
    
    print("🔄 Migration to Medical Embeddings System")
    print("=" * 50)
    
    # Check if old collection exists
    print("\n1. Checking for existing collections...")
    
    old_vector_store = None
    try:
        # Try to load old collection with general embeddings
        import chromadb
        client = chromadb.PersistentClient(path="./data/chroma_db")
        
        collections = client.list_collections()
        print(f"   Found {len(collections)} existing collections:")
        for collection in collections:
            print(f"   - {collection.name}: {collection.count()} documents")
        
        # Check if we have the old collection
        old_collection = None
        for collection in collections:
            if collection.name == "midi_protocols":
                old_collection = collection
                break
        
        if old_collection:
            print(f"   📊 Found old collection 'midi_protocols' with {old_collection.count()} documents")
            
            # Get all documents from old collection
            print("\n2. Extracting documents from old collection...")
            results = old_collection.get(include=['documents', 'metadatas'])
            
            if results['documents']:
                print(f"   ✅ Extracted {len(results['documents'])} documents")
                
                # Initialize new medical vector store
                print("\n3. Initializing new medical vector store...")
                medical_vector_store = VectorStoreManager()
                
                # Process documents with medical chunking
                print("\n4. Re-processing documents with medical embeddings...")
                
                from langchain.schema import Document
                
                processed_count = 0
                for i, (doc_text, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                    try:
                        # Create enhanced chunks with medical processing
                        source_metadata = {
                            "source": metadata.get("source", f"document_{i}"),
                            "data_source": metadata.get("data_source", "protocols"),
                            "content_type": metadata.get("content_type", "clinical_protocol"),
                            "protocol_type": metadata.get("protocol_type", "general")
                        }
                        
                        # Use medical chunking
                        medical_chunks = medical_vector_store.create_medical_chunks(doc_text, source_metadata)
                        
                        # Add to new collection
                        doc_ids = medical_vector_store.add_documents(medical_chunks)
                        processed_count += len(doc_ids)
                        
                        if (i + 1) % 10 == 0:
                            print(f"   📄 Processed {i + 1}/{len(results['documents'])} documents...")
                            
                    except Exception as e:
                        logger.warning(f"Failed to process document {i}: {e}")
                        continue
                
                print(f"\n5. Migration Results:")
                print(f"   ✅ Successfully migrated {processed_count} chunks from {len(results['documents'])} original documents")
                
                # Get stats on new collection
                stats = medical_vector_store.get_collection_stats()
                print(f"   📊 New medical collection contains: {stats.get('total_documents', 'unknown')} documents")
                
                # Test search with medical embeddings
                print("\n6. Testing medical search capabilities...")
                test_queries = [
                    "hormone replacement therapy contraindications",
                    "breast cancer screening protocols",
                    "menopause treatment guidelines"
                ]
                
                for query in test_queries:
                    try:
                        results = medical_vector_store.search_with_scores(query, k=3)
                        print(f"   🔍 '{query}': found {len(results)} relevant chunks")
                        if results:
                            best_score = results[0][1]
                            best_section = results[0][0].metadata.get('clinical_section', 'unknown')
                            print(f"      Top result: score={best_score:.3f}, section={best_section}")
                    except Exception as e:
                        print(f"   ❌ Search failed for '{query}': {e}")
                
                print(f"\n🎉 Migration Complete!")
                print(f"📈 Your Midi RAG system now uses:")
                print(f"   • Medical domain embeddings (PubMedBERT)")
                print(f"   • Clinical-aware chunking")
                print(f"   • Enhanced medical metadata")
                print(f"   • Better clinical concept understanding")
                
            else:
                print("   ⚠️  Old collection is empty")
        else:
            print("   ℹ️  No old collection found. Starting fresh with medical embeddings.")
            
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        logger.error(f"Migration error: {e}")

def compare_embeddings_performance():
    """Compare old vs new embedding performance on sample queries."""
    
    print("\n" + "=" * 50)
    print("🔬 Embeddings Performance Comparison")
    print("=" * 50)
    
    # Sample clinical queries
    test_queries = [
        "What are the contraindications for hormone replacement therapy?",
        "How should patients be monitored on HRT?",
        "What is the recommended dosing for estradiol?",
        "Which patients are eligible for testosterone therapy?",
        "What are the risks of venous thromboembolism with HRT?"
    ]
    
    try:
        # Load medical embeddings
        medical_vector_store = VectorStoreManager()
        
        print("\n🏥 Testing medical domain search relevance...")
        for query in test_queries[:2]:  # Test first 2 queries
            print(f"\n🔍 Query: '{query}'")
            results = medical_vector_store.search_with_scores(query, k=3)
            
            for i, (doc, score) in enumerate(results):
                section = doc.metadata.get('clinical_section', 'unknown')
                concepts = ', '.join(doc.metadata.get('medical_concepts', [])[:3])
                print(f"   {i+1}. Score: {score:.3f} | Section: {section} | Concepts: {concepts}")
                print(f"      Content: {doc.page_content[:100]}...")
                
    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")

if __name__ == "__main__":
    migrate_to_medical_embeddings()
    compare_embeddings_performance()