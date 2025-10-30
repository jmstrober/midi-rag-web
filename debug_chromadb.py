#!/usr/bin/env python3
"""
Debug ChromaDB to see what collections and documents exist
"""

import chromadb
import os
from pathlib import Path

def debug_chromadb():
    # Connect to ChromaDB
    persist_directory = "./data/chroma_db"
    print(f"Connecting to ChromaDB at: {persist_directory}")
    
    try:
        client = chromadb.PersistentClient(path=persist_directory)
        print("✅ Connected to ChromaDB")
        
        # List all collections
        collections = client.list_collections()
        print(f"\n📁 Found {len(collections)} collections:")
        
        for collection in collections:
            print(f"  - {collection.name} (count: {collection.count()})")
            
            # If this is the midi_protocols collection, check some documents
            if collection.name == "midi_protocols":
                print(f"\n🔍 Examining midi_protocols collection:")
                
                # Get a few sample documents
                sample_results = collection.get(limit=3, include=['documents', 'metadatas'])
                
                print(f"Sample documents:")
                for i, (doc, metadata) in enumerate(zip(sample_results['documents'], sample_results['metadatas'])):
                    print(f"\nDocument {i+1}:")
                    print(f"  Content preview: {doc[:200]}...")
                    print(f"  Metadata: {metadata}")
                
                # Test search for breast cancer related content
                print(f"\n🔍 Testing search for 'breast cancer HRT':")
                try:
                    search_results = collection.query(
                        query_texts=["breast cancer HRT family history"],
                        n_results=3,
                        include=['documents', 'metadatas', 'distances']
                    )
                    
                    print(f"Found {len(search_results['documents'][0])} results:")
                    for i, (doc, metadata, distance) in enumerate(zip(
                        search_results['documents'][0], 
                        search_results['metadatas'][0],
                        search_results['distances'][0]
                    )):
                        print(f"\nResult {i+1} (distance: {distance:.3f}):")
                        print(f"  Content: {doc[:300]}...")
                        print(f"  Metadata: {metadata}")
                        
                except Exception as e:
                    print(f"❌ Search failed: {e}")
        
        if not collections:
            print("❌ No collections found!")
            
    except Exception as e:
        print(f"❌ Failed to connect to ChromaDB: {e}")

if __name__ == "__main__":
    debug_chromadb()