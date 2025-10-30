#!/usr/bin/env python3
"""
Quick script to check ChromaDB status
"""
import chromadb

try:
    client = chromadb.PersistentClient(path='./data/chroma_db')
    collections = client.list_collections()
    print(f'Found {len(collections)} collections')
    
    for coll in collections:
        count = coll.count()
        print(f'Collection "{coll.name}": {count} documents')
        
        if count > 0:
            # Sample a few documents
            results = coll.peek(limit=3)
            print(f'Sample IDs: {results.get("ids", [])[:3]}')
            
except Exception as e:
    print(f'Error checking database: {e}')