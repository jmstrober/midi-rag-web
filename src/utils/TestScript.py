#!/usr/bin/env python3
"""
CLI script to ingest clinical protocols into the RAG system.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion_pipeline import IngestionPipeline

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ingestion.log')
        ]
    )

def main():
    """Main function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting protocol ingestion...")
    
    # Initialize pipeline
    pipeline = IngestionPipeline()
    
    # Get current stats
    logger.info("Current stats:")
    stats = pipeline.get_stats()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # Run ingestion
    logger.info("Running ingestion...")
    results = pipeline.ingest_directory()
    
    # Print results
    logger.info("Ingestion results:")
    for key, value in results.items():
        if key != "errors":
            logger.info(f"  {key}: {value}")
    
    if results.get("errors"):
        logger.error("Errors encountered:")
        for error in results["errors"]:
            logger.error(f"  {error}")
    
    # Get updated stats
    logger.info("Updated stats:")
    stats = pipeline.get_stats()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

if __name__ == "__main__":
    main()