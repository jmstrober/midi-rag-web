import logging
import chromadb
from pathlib import Path
from typing import List, Optional, Dict, Any
# Import sentence_transformers only when needed to avoid loading issues
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document
import uuid

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages the vector database for storing and retrieving clinical protocols."""
    
    def __init__(self, persist_directory="./data/chroma_db"):
        """Initialize the VectorStoreManager with enhanced error handling and fallback."""
        self.persist_directory = persist_directory
        self.collection = None
        self.embeddings = None
        self.vectorstore = None
        self.use_fallback = False  # Try embeddings first
        
        # Try to initialize with proper embeddings first
        logger.info("� Attempting to initialize with AI embeddings...")
        
        try:
            self._init_with_embeddings()
            logger.info("✅ AI embeddings system initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize with embeddings: {e}")
            logger.info("🔄 Falling back to ChromaDB text-based search...")
            self.use_fallback = True
            
            if self._init_fallback_with_chromadb():
                logger.info("✅ ChromaDB fallback search system initialized successfully")
            else:
                logger.error("❌ Failed to initialize fallback system")
                raise RuntimeError("Could not initialize any search system")
    
    def _init_with_embeddings(self):
        """Try to initialize with sentence transformers embeddings."""
        try:
            import torch
            
            # Set environment variables to prevent multiprocessing issues AND force offline mode
            import os
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            os.environ['OMP_NUM_THREADS'] = '1'
            
            # Detect if we're running in Streamlit Cloud or locally
            is_streamlit_cloud = (
                'STREAMLIT_SHARING_MODE' in os.environ or 
                'STREAMLIT_SERVER_HEADLESS' in os.environ or
                'HOSTNAME' in os.environ and 'streamlit' in os.environ.get('HOSTNAME', '').lower()
            )
            
            if is_streamlit_cloud:
                # Allow online mode for Streamlit Cloud to download models
                logger.info("🌐 Running in Streamlit Cloud - online mode enabled for model downloads")
                # Set longer timeout for cloud model downloads
                os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'  # Disable progress bars in cloud
                
            else:
                # Force offline mode to prevent any Hugging Face Hub communication (local only)
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['HF_DATASETS_OFFLINE'] = '1'
                logger.info("🔒 Running locally - forced offline mode for Hugging Face models")
            
            from sentence_transformers import SentenceTransformer
            
            # Use the installed LangChain packages that work
            # Try newer langchain_huggingface first, fall back to langchain_community
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                logger.info("✅ Using langchain_huggingface package")
            except ImportError:
                try:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    logger.info("✅ Using langchain_community.embeddings package")
                except ImportError:
                    logger.error("❌ Neither langchain_huggingface nor langchain_community.embeddings available")
                    raise ImportError("No compatible HuggingFace embeddings package found")
            
            from langchain_chroma import Chroma
            
            # DO NOT configure PyTorch threading - this causes "parallel work has started" errors!
            logger.info("⚠️ Skipping PyTorch threading configuration to avoid conflicts")
            
            # Use a smaller, more reliable model with multiple fallback strategies
            model_candidates = [
                "all-MiniLM-L6-v2",           # Primary choice (384 dim)
                "paraphrase-MiniLM-L6-v2",    # Alternative (384 dim) 
                "all-distilroberta-v1",       # Backup (768 dim)
                "paraphrase-distilroberta-base-v1"  # Last resort (768 dim)
            ]
            
            embeddings_initialized = False
            chosen_model = None
            
            for model_name in model_candidates:
                try:
                    logger.info(f"Trying embedding model: {model_name}")
                    
                    # Initialize embeddings with minimal settings to avoid conflicts
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name=model_name
                    )
                    
                    # Test the embeddings with a small example
                    logger.info("Testing embeddings with sample text...")
                    test_result = self.embeddings.embed_query("test query")
                    logger.info(f"✅ Embeddings test successful with {model_name}, vector size: {len(test_result)}")
                    
                    chosen_model = model_name
                    embeddings_initialized = True
                    break
                    
                except Exception as model_error:
                    logger.warning(f"⚠️ Failed to initialize {model_name}: {model_error}")
                    continue
            
            if not embeddings_initialized:
                raise RuntimeError("Failed to initialize any embedding model")
            
            logger.info(f"✅ Successfully initialized embeddings with: {chosen_model}")
            
            # Initialize Chroma vector store
            logger.info(f"Connecting to Chroma database at: {self.persist_directory}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="midi_protocols"  # Connect to the existing collection with data
            )
            
            # Test vector store connection
            collections = self.vectorstore._client.list_collections()
            if collections:
                logger.info(f"✅ Found {len(collections)} collections in vector store")
                # Set up the collection and embedding model for search methods
                self.collection = self.vectorstore._collection
                # Import SentenceTransformer only when embeddings work
                from sentence_transformers import SentenceTransformer
                logger.info(f"Initializing SentenceTransformer {chosen_model}")
                # Use None cache folder to use default system cache location
                cache_folder = None if is_streamlit_cloud else None
                self.embedding_model = SentenceTransformer(chosen_model, cache_folder=cache_folder)
                return True
            else:
                logger.warning("⚠️ No collections found in vector store - may need re-ingestion")
                # Even if empty, set up the objects for potential future use
                self.collection = self.vectorstore._collection
                from sentence_transformers import SentenceTransformer
                logger.info(f"Initializing SentenceTransformer {chosen_model} (empty store)")
                # Use None cache folder to use default system cache location  
                cache_folder = None if is_streamlit_cloud else None
                self.embedding_model = SentenceTransformer(chosen_model, cache_folder=cache_folder)
                return True  # Still valid, just empty
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Failed to initialize embeddings: {error_msg}")
            
            # Check for specific PyTorch errors
            if "parallel work has started" in error_msg.lower():
                logger.error("🔍 Detected PyTorch threading conflict - this is a known issue")
            elif "segmentation fault" in error_msg.lower() or "sigsegv" in error_msg.lower():
                logger.error("🔍 Detected segmentation fault - memory/process issue")
            
            self.embeddings = None
            self.vectorstore = None
            return False
    
    def _init_fallback_with_chromadb(self):
        """Initialize fallback mode but using real ChromaDB data for better results."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Connect directly to ChromaDB without embeddings
            logger.info(f"Connecting directly to ChromaDB at: {self.persist_directory}")
            client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Try to get the midi_protocols collection which has all our data
            collections = client.list_collections()
            logger.info(f"Found {len(collections)} collections in ChromaDB")
            
            midi_collection = None
            for collection in collections:
                if collection.name == "midi_protocols":
                    midi_collection = collection
                    break
            
            if midi_collection:
                logger.info(f"✅ Found midi_protocols collection with {midi_collection.count()} documents")
                
                # Get all documents from the collection for text-based search
                logger.info("Loading all documents for text-based search...")
                all_results = midi_collection.get(
                    include=['documents', 'metadatas']
                )
                
                # Create a simple text-based search using the real ChromaDB documents
                class ChromaDBFallbackStore:
                    def __init__(self, documents, metadatas):
                        self.documents = []
                        for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
                            self.documents.append({
                                'content': doc,
                                'metadata': metadata or {}
                            })
                        logger.info(f"Loaded {len(self.documents)} documents for text search")
                    
                    def search_with_scores(self, query, k=5, filter_dict=None):
                        import re
                        query_words = re.findall(r'\b\w+\b', query.lower())
                        results = []
                        
                        for doc_data in self.documents:
                            content = doc_data['content'].lower()
                            metadata = doc_data['metadata']
                            
                            # Apply filters if provided
                            if filter_dict:
                                skip = False
                                for key, value in filter_dict.items():
                                    if metadata.get(key) != value:
                                        skip = True
                                        break
                                if skip:
                                    continue
                            
                            # Calculate relevance score based on keyword matches
                            score = 0
                            for word in query_words:
                                if word in content:
                                    score += content.count(word) * 0.1
                            
                            # Boost for important medical terms
                            medical_terms = ['breast cancer', 'hrt', 'hormone', 'menopause', 'protocol']
                            for term in medical_terms:
                                if term in query.lower() and term in content:
                                    score += 1.0
                            
                            if score > 0:
                                # Create a mock document
                                class MockDoc:
                                    def __init__(self, content, metadata):
                                        self.page_content = doc_data['content']  # Use original content, not lowercased
                                        self.metadata = metadata
                                results.append((MockDoc(doc_data['content'], metadata), score))
                        
                        return sorted(results, key=lambda x: x[1], reverse=True)[:k]
                    
                    def get_collection_stats(self):
                        return {'document_count': len(self.documents), 'type': 'chromadb_fallback'}
                
                self.fallback_store = ChromaDBFallbackStore(all_results['documents'], all_results['metadatas'])
                logger.info("✅ ChromaDB fallback text search initialized")
                return True
            else:
                logger.warning("⚠️ midi_protocols collection not found, using minimal fallback")
                return self._init_fallback()
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize ChromaDB fallback: {e}")
            logger.info("Falling back to minimal text search...")
            return self._init_fallback()

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
                    # Include comprehensive clinical knowledge for fallback
                    self.documents = [
                        {
                            'content': 'Hormone replacement therapy (HRT) is used to treat menopausal symptoms including hot flashes, night sweats, and vaginal dryness. Benefits include symptom relief and bone protection. Risks include increased risk of blood clots and breast cancer in some women. Contraindications include active breast cancer, undiagnosed vaginal bleeding, and active blood clots.',
                            'metadata': {'source': 'HRT Protocol', 'data_source': 'protocols', 'protocol_type': 'menopause'}
                        },
                        {
                            'content': 'Hot flashes are sudden feelings of warmth, often accompanied by sweating and rapid heartbeat. Treatment options include hormone therapy (estrogen with or without progesterone), selective serotonin reuptake inhibitors (SSRIs) like paroxetine, gabapentin, clonidine, and lifestyle modifications including avoiding triggers, staying cool, and stress management.',
                            'metadata': {'source': 'Hot Flash Management', 'data_source': 'protocols', 'protocol_type': 'menopause'}
                        },
                        {
                            'content': 'Weight management during menopause can be challenging due to hormonal changes that slow metabolism and increase abdominal fat. Strategies include regular exercise (both cardio and strength training), balanced nutrition with adequate protein, adequate sleep, stress management, and intermittent fasting. Some patients may benefit from weight loss medications like semaglutide or liraglutide.',
                            'metadata': {'source': 'Weight Management Protocol', 'data_source': 'protocols', 'protocol_type': 'weight_management'}
                        },
                        {
                            'content': 'Sleep disturbances are common during menopause due to hot flashes, night sweats, and hormonal changes. Treatment approaches include sleep hygiene education, melatonin 1-3mg before bedtime, cognitive behavioral therapy for insomnia (CBT-I), addressing underlying hot flashes with hormone therapy, and considering sleep aids like trazodone or zolpidem for short-term use.',
                            'metadata': {'source': 'Sleep Protocol', 'data_source': 'protocols', 'protocol_type': 'sleep'}
                        },
                        {
                            'content': 'Vaginal dryness and painful intercourse (dyspareunia) are common symptoms of menopause caused by decreased estrogen leading to vaginal atrophy. Treatment options include vaginal moisturizers (used regularly), personal lubricants (used during intercourse), and low-dose vaginal estrogen therapy (creams, tablets, or rings). Vaginal estrogen is generally safe even for breast cancer survivors.',
                            'metadata': {'source': 'Sexual Health Protocol', 'data_source': 'protocols', 'protocol_type': 'sexual_health'}
                        },
                        {
                            'content': 'Perimenopause is the transitional period before menopause, typically lasting 4-8 years. Symptoms include irregular periods, hot flashes, mood changes, sleep disturbances, and cognitive changes. Hormone levels fluctuate unpredictably. Treatment may include low-dose birth control pills for younger women or hormone therapy for severe symptoms.',
                            'metadata': {'source': 'Perimenopause Protocol', 'data_source': 'protocols', 'protocol_type': 'menopause'}
                        },
                        {
                            'content': 'Mood disorders during menopause include depression, anxiety, and irritability due to hormonal fluctuations. Risk factors include history of depression, severe menopausal symptoms, and psychosocial stressors. Treatment options include hormone therapy, antidepressants (SSRIs/SNRIs), counseling, stress management, and lifestyle modifications.',
                            'metadata': {'source': 'Mood Disorders Protocol', 'data_source': 'protocols', 'protocol_type': 'mood'}
                        },
                        {
                            'content': 'Bone health becomes critical during menopause due to decreased estrogen leading to accelerated bone loss. Prevention strategies include adequate calcium (1200mg daily) and vitamin D (800-1000 IU daily), weight-bearing exercise, smoking cessation, and limiting alcohol. Hormone therapy provides bone protection. DEXA scans should be performed at menopause and every 1-2 years thereafter.',
                            'metadata': {'source': 'Bone Health Protocol', 'data_source': 'protocols', 'protocol_type': 'bone_health'}
                        }
                    ]
                    logger.info(f"Using comprehensive inline fallback store with {len(self.documents)} clinical documents")
                
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