import logging
import chromadb
from pathlib import Path
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document
import uuid

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages the vector database for storing and retrieving clinical protocols."""
    
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.use_fallback = False
        
        # Check if we're running on Streamlit Cloud (which has PyTorch issues)
        import os
        is_streamlit_cloud = os.path.exists("/mount/src") or "STREAMLIT" in os.environ
        
        if is_streamlit_cloud:
            logger.info("Detected Streamlit Cloud environment - using text-based fallback")
            self._init_fallback()
        else:
            # Try to initialize with embeddings first for local environments
            try:
                self._init_with_embeddings()
            except Exception as e:
                logger.warning(f"Failed to initialize with embeddings: {e}")
                logger.info("Falling back to simple text-based search...")
                self._init_fallback()
    
    def _init_with_embeddings(self):
        """Initialize with sentence transformers embeddings."""
        # Initialize embeddings model with explicit device and dtype settings
        try:
            import torch
            # Force CPU and disable meta device usage
            torch.set_default_device('cpu')
            torch.set_default_dtype(torch.float32)
            
            # Load model with explicit parameters to avoid meta tensor issues
            self.embedding_model = SentenceTransformer(
                'all-MiniLM-L6-v2', 
                device='cpu',
                cache_folder=None,  # Disable caching to avoid corruption
                use_auth_token=False
            )
            logger.info("✅ Embedding model loaded successfully on CPU")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise e
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )
        
        # Get or create collection
        self.collection_name = "midi_protocols"
        try:
            self.collection = self.client.get_collection(self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except chromadb.errors.NotFoundError:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Midi clinical protocols for RAG system"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
        
        logger.info(f"Vector store initialized at {self.persist_directory}")
    
    def _init_fallback(self):
        """Initialize with simple text-based fallback."""
        import sys
        from pathlib import Path
        
        # Add the utils directory to the path for import
        utils_dir = Path(__file__).parent
        if str(utils_dir) not in sys.path:
            sys.path.append(str(utils_dir))
        
        try:
            from SimpleVectorStore import SimpleVectorStore
        except ImportError:
            # If that fails, define a minimal fallback inline
            class SimpleVectorStore:
                def __init__(self):
                    # Include some basic clinical knowledge for fallback
                    self.documents = [
                        {
                            'content': 'Hormone replacement therapy (HRT) is used to treat menopausal symptoms including hot flashes, night sweats, and vaginal dryness. Benefits include symptom relief and bone protection. Risks include increased risk of blood clots and breast cancer in some women.',
                            'metadata': {'source': 'HRT Protocol', 'data_source': 'protocols', 'protocol_type': 'menopause'}
                        },
                        {
                            'content': 'Hot flashes are sudden feelings of warmth, often accompanied by sweating and rapid heartbeat. Treatment options include hormone therapy, selective serotonin reuptake inhibitors (SSRIs), gabapentin, and lifestyle modifications.',
                            'metadata': {'source': 'Hot Flash Management', 'data_source': 'protocols', 'protocol_type': 'menopause'}
                        },
                        {
                            'content': 'Weight management during menopause can be challenging due to hormonal changes. Strategies include regular exercise, balanced nutrition, adequate sleep, and stress management. Some patients may benefit from weight loss medications.',
                            'metadata': {'source': 'Weight Management Protocol', 'data_source': 'protocols', 'protocol_type': 'weight_management'}
                        },
                        {
                            'content': 'Sleep disturbances are common during menopause. Treatment approaches include sleep hygiene, melatonin, cognitive behavioral therapy for insomnia, and addressing underlying hot flashes.',
                            'metadata': {'source': 'Sleep Protocol', 'data_source': 'protocols', 'protocol_type': 'sleep'}
                        },
                        {
                            'content': 'Vaginal dryness and painful intercourse are common symptoms of menopause caused by decreased estrogen. Treatment options include vaginal moisturizers, lubricants, and low-dose vaginal estrogen therapy.',
                            'metadata': {'source': 'Sexual Health Protocol', 'data_source': 'protocols', 'protocol_type': 'sexual_health'}
                        }
                    ]
                    logger.info(f"Using minimal inline fallback store with {len(self.documents)} sample documents")
                
                def search_with_scores(self, query, k=5, filter_dict=None):
                    # Improved keyword matching
                    results = []
                    query_words = set(query.lower().split())
                    
                    for doc in self.documents:
                        content = doc.get('content', '').lower()
                        content_words = set(content.split())
                        
                        # Calculate basic similarity score
                        common_words = query_words.intersection(content_words)
                        score = len(common_words) / len(query_words) if query_words else 0
                        
                        # Boost score for exact phrase matches
                        if query.lower() in content:
                            score += 0.5
                        
                        # Apply filters if provided
                        if filter_dict:
                            metadata = doc.get('metadata', {})
                            skip = False
                            for key, value in filter_dict.items():
                                if metadata.get(key) != value:
                                    skip = True
                                    break
                            if skip:
                                continue
                        
                        if score > 0:
                            # Create a mock document
                            class MockDoc:
                                def __init__(self, content, metadata):
                                    self.page_content = content
                                    self.metadata = metadata
                            results.append((MockDoc(doc['content'], doc.get('metadata', {})), score))
                    
                    return sorted(results, key=lambda x: x[1], reverse=True)[:k]
                
                def get_collection_stats(self):
                    return {'document_count': len(self.documents), 'type': 'minimal_fallback'}
        
        self.use_fallback = True
        self.fallback_store = SimpleVectorStore()
        self.embedding_model = None
        logger.info("✅ Fallback text-based store initialized")
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        try:
            if not documents:
                logger.warning("No documents to add")
                return []
            
            # Prepare data for ChromaDB
            doc_ids = []
            texts = []
            metadatas = []
            
            for doc in documents:
                # Generate unique ID if not provided
                doc_id = doc.metadata.get('chunk_id', str(uuid.uuid4()))
                doc_ids.append(doc_id)
                texts.append(doc.page_content)
                
                # Clean metadata for ChromaDB (ensure all values are strings, numbers, or booleans)
                clean_metadata = {}
                for key, value in doc.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        clean_metadata[key] = value
                    else:
                        clean_metadata[key] = str(value)
                
                metadatas.append(clean_metadata)
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts).tolist()
            
            # Add to collection
            self.collection.add(
                ids=doc_ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings
            )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            return []
    
    def search_similar(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[Document]:
        """Search for similar documents."""
        try:
            # Check if collection is empty
            if self.collection.count() == 0:
                logger.warning("Vector store is empty. No documents to search.")
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter_dict
            )
            
            # Convert results to LangChain Document format
            documents = []
            if results['documents'] and results['documents'][0]:
                for i, (doc_text, metadata) in enumerate(zip(
                    results['documents'][0], 
                    results['metadatas'][0]
                )):
                    documents.append(Document(
                        page_content=doc_text,
                        metadata=metadata
                    ))
            
            logger.info(f"Found {len(documents)} similar documents for query")
            return documents
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def search_with_scores(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[tuple]:
        """Search for similar documents with similarity scores."""
        # Use fallback if embeddings failed
        if self.use_fallback:
            return self.fallback_store.search_with_scores(query, k, filter_dict)
        
        try:
            # Check if collection is empty
            if self.collection.count() == 0:
                logger.warning("Vector store is empty. No documents to search.")
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter_dict
            )
            
            # Convert results to (Document, score) tuples
            documents_with_scores = []
            if results['documents'] and results['documents'][0]:
                for i, (doc_text, metadata, distance) in enumerate(zip(
                    results['documents'][0], 
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (higher is more similar)
                    similarity_score = 1 - distance
                    doc = Document(page_content=doc_text, metadata=metadata)
                    documents_with_scores.append((doc, similarity_score))
            
            logger.info(f"Found {len(documents_with_scores)} similar documents with scores")
            return documents_with_scores
            
        except Exception as e:
            logger.error(f"Error searching vector store with scores: {str(e)}")
            # Try fallback on error
            if not self.use_fallback:
                logger.info("Attempting fallback search...")
                try:
                    self._init_fallback()
                    return self.fallback_store.search_with_scores(query, k, filter_dict)
                except Exception as e2:
                    logger.error(f"Fallback search also failed: {e2}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "persist_directory": str(self.persist_directory)
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}
    
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents by their IDs."""
        try:
            self.collection.delete(ids=doc_ids)
            logger.info(f"Deleted {len(doc_ids)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False