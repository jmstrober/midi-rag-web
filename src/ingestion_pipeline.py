import logging
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime
import sys

# Add utils to path
sys.path.append(str(Path(__file__).parent / 'utils'))

from document_processor import DocumentProcessor
from VectorStoreManager import VectorStoreManager

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """Main pipeline for ingesting clinical protocols and other documents into the RAG system."""
    
    def __init__(self, base_data_directory: str = "./data"):
        self.base_data_directory = Path(base_data_directory)
        self.base_data_directory.mkdir(parents=True, exist_ok=True)
        
        # Define data source directories
        self.data_sources = {
            "protocols": self.base_data_directory / "protocols",
            "midi_blog_posts": self.base_data_directory / "midi_blog_posts", 
            "midi_zendesk_articles": self.base_data_directory / "midi_zendesk_articles"
        }
        
        # Create directories if they don't exist
        for source_name, source_path in self.data_sources.items():
            source_path.mkdir(parents=True, exist_ok=True)
        
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStoreManager()
        
        # Track processed files (now per data source)
        self.processed_files_logs = {}
        self.processed_files = {}
        
        for source_name, source_path in self.data_sources.items():
            log_file = source_path / "processed_files.json"
            self.processed_files_logs[source_name] = log_file
            self.processed_files[source_name] = self._load_processed_files(source_name)
        
        logger.info("Enhanced ingestion pipeline initialized with multiple data sources")
        logger.info(f"Data sources: {list(self.data_sources.keys())}")
    
    def ingest_all_sources(self, force_reingest: bool = False) -> Dict[str, Any]:
        """Ingest all files from all data sources."""
        results = {
            "success": True,
            "sources": {},
            "total_stats": {
                "total_files": 0,
                "processed_files": 0,
                "skipped_files": 0,
                "failed_files": 0,
                "total_chunks": 0,
                "errors": []
            }
        }
        
        for source_name, source_path in self.data_sources.items():
            logger.info(f"Processing data source: {source_name}")
            source_result = self.ingest_data_source(source_name, force_reingest=force_reingest)
            results["sources"][source_name] = source_result
            
            # Aggregate totals
            for key in ["total_files", "processed_files", "skipped_files", "failed_files", "total_chunks"]:
                results["total_stats"][key] += source_result.get(key, 0)
            
            results["total_stats"]["errors"].extend(source_result.get("errors", []))
            
            if not source_result.get("success", False):
                results["success"] = False
        
        logger.info(f"All sources ingestion complete: {results['total_stats']}")
        return results
    
    def ingest_data_source(self, source_name: str, force_reingest: bool = False) -> Dict[str, Any]:
        """Ingest all supported files from a specific data source."""
        if source_name not in self.data_sources:
            error_msg = f"Unknown data source: {source_name}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        source_path = self.data_sources[source_name]
        
        if not source_path.exists():
            logger.warning(f"Data source directory does not exist: {source_path}")
            return {"success": True, "total_files": 0, "processed_files": 0, "skipped_files": 0, "failed_files": 0, "total_chunks": 0, "errors": []}
        
        # Find all supported files
        supported_extensions = ['.pdf', '.docx', '.doc', '.md', '.markdown', '.txt']
        files_to_process = []
        
        for ext in supported_extensions:
            files_to_process.extend(source_path.glob(f"*{ext}"))
            files_to_process.extend(source_path.glob(f"**/*{ext}"))  # Include subdirectories
        
        logger.info(f"Found {len(files_to_process)} files in {source_name}")
        
        # Process files
        results = {
            "success": True,
            "source_name": source_name,
            "source_path": str(source_path),
            "total_files": len(files_to_process),
            "processed_files": 0,
            "skipped_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "errors": []
        }
        
        for file_path in files_to_process:
            try:
                result = self.ingest_file(file_path, source_name, force_reingest=force_reingest)
                if result["success"]:
                    results["processed_files"] += 1
                    results["total_chunks"] += result["chunks_added"]
                else:
                    if result.get("skipped"):
                        results["skipped_files"] += 1
                    else:
                        results["failed_files"] += 1
                        results["errors"].append(f"{file_path}: {result.get('error', 'Unknown error')}")
                        
            except Exception as e:
                results["failed_files"] += 1
                results["errors"].append(f"{file_path}: {str(e)}")
                logger.error(f"Unexpected error processing {file_path}: {str(e)}")
        
        # Save processed files log for this source
        self._save_processed_files(source_name)
        
        logger.info(f"Source {source_name} ingestion complete: {results}")
        return results
    
    def ingest_file(self, file_path: Path, source_name: str, force_reingest: bool = False) -> Dict[str, Any]:
        """Ingest a single file from a specific data source."""
        file_path = Path(file_path)
        
        # Check if file was already processed and hasn't changed
        file_hash = self._get_file_hash(file_path)
        if not force_reingest and self._is_file_processed(file_path, file_hash, source_name):
            logger.info(f"File already processed: {file_path}")
            return {"success": True, "skipped": True, "chunks_added": 0}
        
        # Process the document with enhanced metadata
        documents = self.document_processor.process_file(file_path)
        
        if not documents:
            error_msg = f"No documents extracted from {file_path}"
            logger.warning(error_msg)
            return {"success": False, "error": error_msg, "chunks_added": 0}
        
        # Enhance document metadata with source information
        for doc in documents:
            if not hasattr(doc, 'metadata'):
                doc.metadata = {}
            doc.metadata.update({
                "data_source": source_name,
                "source_file": str(file_path),
                "content_type": self._get_content_type(source_name, file_path)
            })
        
        # Add to vector store
        doc_ids = self.vector_store.add_documents(documents)
        
        if not doc_ids:
            error_msg = f"Failed to add documents to vector store for {file_path}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "chunks_added": 0}
        
        # Update processed files log for this source
        self.processed_files[source_name][str(file_path)] = {
            "processed_at": datetime.now().isoformat(),
            "file_hash": file_hash,
            "chunks_count": len(documents),
            "doc_ids": doc_ids,
            "content_type": self._get_content_type(source_name, file_path)
        }
        
        logger.info(f"Successfully ingested {file_path} ({source_name}): {len(documents)} chunks")
        return {"success": True, "chunks_added": len(documents)}
    
    def _get_content_type(self, source_name: str, file_path: Path) -> str:
        """Determine content type based on source and file."""
        content_types = {
            "protocols": "clinical_protocol",
            "midi_blog_posts": "blog_post", 
            "midi_zendesk_articles": "support_article"
        }
        return content_types.get(source_name, "unknown")
        
        # Add to vector store
        doc_ids = self.vector_store.add_documents(documents)
        
        if not doc_ids:
            error_msg = f"Failed to add documents to vector store for {file_path}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "chunks_added": 0}
        
        # Update processed files log
        self.processed_files[str(file_path)] = {
            "processed_at": datetime.now().isoformat(),
            "file_hash": file_hash,
            "chunks_count": len(documents),
            "doc_ids": doc_ids
        }
        
        logger.info(f"Successfully ingested {file_path}: {len(documents)} chunks")
        return {"success": True, "chunks_added": len(documents)}
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file contents."""
        import hashlib
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _is_file_processed(self, file_path: Path, current_hash: str, source_name: str) -> bool:
        """Check if file was already processed and hasn't changed."""
        file_key = str(file_path)
        source_files = self.processed_files.get(source_name, {})
        
        if file_key not in source_files:
            return False
        
        stored_hash = source_files[file_key].get("file_hash")
        return stored_hash == current_hash
    
    def _load_processed_files(self, source_name: str) -> Dict[str, Any]:
        """Load the log of processed files for a specific source."""
        log_file = self.processed_files_logs[source_name]
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load processed files log for {source_name}: {str(e)}")
        return {}
    
    def _save_processed_files(self, source_name: str) -> None:
        """Save the log of processed files for a specific source."""
        log_file = self.processed_files_logs[source_name]
        try:
            with open(log_file, 'w') as f:
                json.dump(self.processed_files[source_name], f, indent=2)
        except Exception as e:
            logger.error(f"Could not save processed files log for {source_name}: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the ingestion pipeline."""
        vector_stats = self.vector_store.get_collection_stats()
        
        source_stats = {}
        total_processed_files = 0
        
        for source_name, processed_files in self.processed_files.items():
            source_stats[source_name] = {
                "processed_files_count": len(processed_files),
                "directory": str(self.data_sources[source_name]),
                "last_processed": max(
                    [info.get("processed_at", "") for info in processed_files.values()],
                    default="Never"
                )
            }
            total_processed_files += len(processed_files)
        
        return {
            "vector_store": vector_stats,
            "data_sources": source_stats,
            "total_processed_files": total_processed_files,
            "base_directory": str(self.base_data_directory)
        }