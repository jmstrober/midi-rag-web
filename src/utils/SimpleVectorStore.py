"""
Simple fallback vector store that uses basic text matching instead of embeddings
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
import json
import re

logger = logging.getLogger(__name__)

class SimpleVectorStore:
    """A simple fallback vector store using text matching when embeddings fail."""
    
    def __init__(self, persist_directory: str = "./data/simple_store"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.documents_file = self.persist_directory / "documents.json"
        
        # Load existing documents
        self.documents = []
        if self.documents_file.exists():
            try:
                with open(self.documents_file, 'r') as f:
                    self.documents = json.load(f)
                logger.info(f"Loaded {len(self.documents)} documents from simple store")
            except Exception as e:
                logger.error(f"Error loading documents: {e}")
                self.documents = []
        
        logger.info(f"Simple vector store initialized with {len(self.documents)} documents")
    
    def add_documents(self, documents):
        """Add documents to the simple store."""
        for doc in documents:
            doc_dict = {
                'content': doc.page_content,
                'metadata': doc.metadata
            }
            self.documents.append(doc_dict)
        
        # Save to file
        try:
            with open(self.documents_file, 'w') as f:
                json.dump(self.documents, f, indent=2)
            logger.info(f"Saved {len(self.documents)} documents to simple store")
        except Exception as e:
            logger.error(f"Error saving documents: {e}")
    
    def search_with_scores(self, query: str, k: int = 5, filter_dict=None):
        """Simple text-based search."""
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))
        
        scored_docs = []
        
        for doc in self.documents:
            content_lower = doc['content'].lower()
            content_words = set(re.findall(r'\w+', content_lower))
            
            # Simple scoring based on word matches
            common_words = query_words.intersection(content_words)
            score = len(common_words) / len(query_words) if query_words else 0
            
            # Boost score for exact phrase matches
            if query_lower in content_lower:
                score += 0.5
            
            # Apply filters if provided
            if filter_dict:
                skip = False
                for key, value in filter_dict.items():
                    if doc['metadata'].get(key) != value:
                        skip = True
                        break
                if skip:
                    continue
            
            if score > 0:
                # Create a document-like object
                class SimpleDoc:
                    def __init__(self, content, metadata):
                        self.page_content = content
                        self.metadata = metadata
                
                scored_docs.append((SimpleDoc(doc['content'], doc['metadata']), score))
        
        # Sort by score and return top k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:k]
    
    def get_collection_stats(self):
        """Get statistics about the collection."""
        return {
            'document_count': len(self.documents),
            'type': 'simple_text_store'
        }