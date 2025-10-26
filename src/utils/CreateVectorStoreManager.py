# Manages a vector database for storing and retrieving clinical protocols using ChromaDB and LangChain
import logging
from pathlib import Path
from typing import List, Optional
import chromadb
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages the vector database for storing and retrieving clinical protocols."""
    
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"  # Good balance of speed and quality
        )
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize or get collection
        self.collection_name = "midi_protocols"
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_directory)
        )
        
        logger.info(f"Vector store initialized at {self.persist_directory}")
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        try:
            if not documents:
                logger.warning("No documents to add")
                return []
            
            # Add documents to vector store
            doc_ids = self.vectorstore.add_documents(documents)
            
            # Persist the changes
            self.vectorstore.persist()
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            return []
    
    def search_similar(self, query: str, k: int = 5, filter_dict: Optional[dict] = None) -> List[Document]:
        """Search for similar documents."""
        try:
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )
            logger.info(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def search_with_scores(self, query: str, k: int = 5, filter_dict: Optional[dict] = None) -> List[tuple]:
        """Search for similar documents with similarity scores."""
        try:
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )
            logger.info(f"Found {len(results)} similar documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store with scores: {str(e)}")
            return []
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the collection."""
        try:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            
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
            self.vectorstore.delete(doc_ids)
            self.vectorstore.persist()
            logger.info(f"Deleted {len(doc_ids)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False