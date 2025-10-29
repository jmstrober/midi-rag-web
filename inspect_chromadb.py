#!/usr/bin/env python3
"""
Check ChromaDB collections and contents
"""

import chromadb
from chromadb.config import Settings

def inspect_chromadb():
    print("🔍 Inspecting ChromaDB...")
    
    # Connect to the existing ChromaDB
    client = chromadb.PersistentClient(path="./data/chroma_db")
    
    # List all collections
    collections = client.list_collections()
    print(f"📚 Found {len(collections)} collections:")
    
    for i, collection in enumerate(collections):
        print(f"  {i+1}. Name: {collection.name}")
        print(f"     ID: {collection.id}")
        
        # Get collection details
        count = collection.count()
        print(f"     Documents: {count}")
        
        if count > 0:
            # Get a few sample documents
            results = collection.get(limit=3)
            print(f"     Sample documents:")
            for j, doc_id in enumerate(results['ids'][:2]):
                metadata = results['metadatas'][j] if results['metadatas'] else {}
                doc_preview = results['documents'][j][:100] if results['documents'] else "No content"
                print(f"       - ID: {doc_id}")
                print(f"         Content: {doc_preview}...")
                print(f"         Metadata: {metadata}")
        print()

if __name__ == "__main__":
    inspect_chromadb()