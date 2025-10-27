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
    """Manages the vector database for storing and retrieving clinical protocols with medical domain embeddings."""
    
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize medical embeddings model for better clinical domain understanding
        self.embedding_model = SentenceTransformer('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
        logger.info("Initialized medical domain embedding model: BiomedNLP-PubMedBERT")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )
        
        # Get or create collection with medical-specific metadata
        self.collection_name = "midi_protocols_medical"
        try:
            self.collection = self.client.get_collection(self.collection_name)
            logger.info(f"Loaded existing medical collection: {self.collection_name}")
        except chromadb.errors.NotFoundError:
            # Collection doesn't exist, create it with medical-specific configuration
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Midi clinical protocols with medical domain embeddings",
                    "embedding_model": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                    "domain": "clinical_medicine",
                    "chunk_strategy": "medical_semantic"
                }
            )
            logger.info(f"Created new medical collection: {self.collection_name}")
        
        logger.info(f"Medical vector store initialized at {self.persist_directory}")
    
    def create_medical_chunks(self, text: str, source_metadata: Dict[str, Any]) -> List[Document]:
        """Create medically-aware chunks from clinical documents."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Medical-specific separators that preserve clinical meaning
        medical_separators = [
            "\n\n## ",          # Protocol sections
            "\n\n### ",         # Subsections
            "\n\n#### ",        # Sub-subsections
            "\n\nBACKGROund:",  # Common clinical sections
            "\n\nCRITERIA:",
            "\n\nCONTRAINDICATION",
            "\n\nINDICATION",
            "\n\nPROCEDURE:",
            "\n\nDOSING:",
            "\n\nMONITORING:",
            "\n\n• ",           # Bullet points
            "\n\n1. ",          # Numbered lists
            "\n\n",             # Paragraph breaks
            "\n",               # Line breaks
            ". ",               # Sentence breaks
            " "                 # Word breaks
        ]
        
        # Use larger chunks for clinical protocols (more context)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,           # Increased from 1000 for better clinical context
            chunk_overlap=300,         # Increased overlap to preserve clinical relationships
            separators=medical_separators,
            length_function=len
        )
        
        # Split the text
        chunks = text_splitter.split_text(text)
        
        # Create Document objects with enhanced metadata
        documents = []
        for i, chunk in enumerate(chunks):
            # Detect medical concepts in the chunk
            medical_concepts = self._extract_medical_concepts(chunk)
            
            chunk_metadata = {
                **source_metadata,
                "chunk_index": i,
                "chunk_id": f"{source_metadata.get('source', 'unknown')}_{i}",
                "chunk_size": len(chunk),
                "medical_concepts": medical_concepts.get("concepts", []),
                "clinical_section": medical_concepts.get("section_type", "general"),
                "contains_contraindications": "contraindication" in chunk.lower(),
                "contains_dosing": any(term in chunk.lower() for term in ["mg", "dose", "dosing", "administr"]),
                "contains_criteria": "criteria" in chunk.lower() or "eligible" in chunk.lower(),
            }
            
            documents.append(Document(
                page_content=chunk,
                metadata=chunk_metadata
            ))
        
        logger.info(f"Created {len(documents)} medical chunks from source: {source_metadata.get('source', 'unknown')}")
        return documents
    
    def _extract_medical_concepts(self, text: str) -> Dict[str, Any]:
        """Extract basic medical concepts and section types from text."""
        text_lower = text.lower()
        
        # Detect section types
        section_type = "general"
        if any(term in text_lower for term in ["background", "overview", "introduction"]):
            section_type = "background"
        elif any(term in text_lower for term in ["criteria", "eligible", "indication"]):
            section_type = "criteria"
        elif any(term in text_lower for term in ["contraindication", "caution", "warning"]):
            section_type = "contraindications"
        elif any(term in text_lower for term in ["procedure", "protocol", "treatment"]):
            section_type = "procedure"
        elif any(term in text_lower for term in ["monitor", "follow", "assess"]):
            section_type = "monitoring"
        elif any(term in text_lower for term in ["dose", "dosing", "administr", "mg", "mcg"]):
            section_type = "dosing"
        
        # Extract basic medical concepts
        medical_terms = []
        clinical_keywords = [
            "hormone", "estrogen", "progesterone", "testosterone",
            "menopause", "perimenopause", "postmenopause",
            "thyroid", "diabetes", "cardiovascular", "breast cancer",
            "contraindication", "indication", "monitoring", "assessment",
            "protocol", "guideline", "recommendation", "criteria"
        ]
        
        for keyword in clinical_keywords:
            if keyword in text_lower:
                medical_terms.append(keyword)
        
        return {
            "section_type": section_type,
            "concepts": medical_terms
        }
    
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