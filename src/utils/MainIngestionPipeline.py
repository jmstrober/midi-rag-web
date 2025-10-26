import logging
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """Main pipeline for ingesting clinical protocols into the RAG system."""
    
    def __init__(self, protocols_directory: str = "./data/protocols"):
        self.protocols_directory = Path(protocols_directory)
        self.protocols_directory.mkdir(parents=True, exist_ok=True)
        
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStoreManager()
        
        # Track processed files
        self.processed_files_log = self.protocols_directory / "processed_files.json"
        self.processed_files = self._load_processed_files()
        
        logger.info("Ingestion pipeline initialized")
    
    def ingest_directory(self, directory_path: str = None) -> Dict[str, Any]:
        """Ingest all supported files from a directory."""
        if directory_path is None:
            directory_path = self.protocols_directory
        else:
            directory_path = Path(directory_path)
        
        if not directory_path.exists():
            logger.error(f"Directory does not exist: {directory_path}")
            return {"success": False, "error": "Directory not found"}
        
        # Find all supported files
        supported_extensions = ['.pdf', '.docx', '.doc', '.md', '.markdown', '.txt']
        files_to_process = []
        
        for ext in supported_extensions:
            files_to_process.extend(directory_path.glob(f"*{ext}"))
            files_to_process.extend(directory_path.glob(f"**/*{ext}"))  # Include subdirectories
        
        logger.info(f"Found {len(files_to_process)} files to process")
        
        # Process files
        results = {
            "success": True,
            "total_files": len(files_to_process),
            "processed_files": 0,
            "skipped_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "errors": []
        }
        
        for file_path in files_to_process:
            try:
                result = self.ingest_file(file_path)
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
        
        # Save processed files log
        self._save_processed_files()
        
        logger.info(f"Ingestion complete: {results}")
        return results
    
    def ingest_file(self, file_path: Path) -> Dict[str, Any]:
        """Ingest a single file."""
        file_path = Path(file_path)
        
        # Check if file was already processed and hasn't changed
        file_hash = self._get_file_hash(file_path)
        if self._is_file_processed(file_path, file_hash):
            logger.info(f"File already processed: {file_path}")
            return {"success": True, "skipped": True, "chunks_added": 0}
        
        # Process the document
        documents = self.document_processor.process_file(file_path)
        
        if not documents:
            error_msg = f"No documents extracted from {file_path}"
            logger.warning(error_msg)
            return {"success": False, "error": error_msg, "chunks_added": 0}
        
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
    
    def _is_file_processed(self, file_path: Path, current_hash: str) -> bool:
        """Check if file was already processed and hasn't changed."""
        file_key = str(file_path)
        if file_key not in self.processed_files:
            return False
        
        stored_hash = self.processed_files[file_key].get("file_hash")
        return stored_hash == current_hash
    
    def _load_processed_files(self) -> Dict[str, Any]:
        """Load the log of processed files."""
        if self.processed_files_log.exists():
            try:
                with open(self.processed_files_log, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load processed files log: {str(e)}")
        return {}
    
    def _save_processed_files(self) -> None:
        """Save the log of processed files."""
        try:
            with open(self.processed_files_log, 'w') as f:
                json.dump(self.processed_files, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save processed files log: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the ingestion pipeline."""
        vector_stats = self.vector_store.get_collection_stats()
        
        return {
            "vector_store": vector_stats,
            "processed_files_count": len(self.processed_files),
            "protocols_directory": str(self.protocols_directory),
            "last_processed": max(
                [info.get("processed_at", "") for info in self.processed_files.values()],
                default="Never"
            )
        }