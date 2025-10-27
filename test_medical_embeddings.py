#!/usr/bin/env python3
"""
Test script for the new medical embeddings system
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.VectorStoreManager import VectorStoreManager
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_medical_embeddings():
    """Test the new medical embedding model and chunking strategy."""
    
    print("🧪 Testing Medical Embeddings System")
    print("=" * 50)
    
    # Initialize the medical vector store
    print("\n1. Initializing Medical VectorStoreManager...")
    try:
        vector_store = VectorStoreManager()
        print("✅ Medical vector store initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize vector store: {e}")
        return
    
    # Test medical chunking
    print("\n2. Testing Medical Document Chunking...")
    
    sample_clinical_text = """
    ## CLINICAL PROTOCOL: Hormone Replacement Therapy in Menopause
    
    ### BACKGROUND:
    Hormone replacement therapy (HRT) is indicated for the treatment of moderate to severe vasomotor symptoms 
    associated with menopause when benefits outweigh risks for the individual patient.
    
    ### INDICATIONS:
    - Moderate to severe vasomotor symptoms (hot flashes, night sweats)
    - Genitourinary syndrome of menopause
    - Prevention of osteoporosis in high-risk patients
    
    ### CONTRAINDICATIONS:
    - Active or history of breast cancer
    - Active or history of endometrial cancer
    - Active venous thromboembolism
    - Active cardiovascular disease
    - Unexplained vaginal bleeding
    
    ### DOSING:
    Initial estradiol dose: 0.5-1.0 mg daily
    Titrate based on symptom response and tolerability
    Add progestin if uterus present: micronized progesterone 100-200 mg daily
    
    ### MONITORING:
    - Baseline mammogram and breast exam
    - Annual breast cancer screening
    - Monitor for signs of thromboembolism
    - Reassess benefits and risks annually
    """
    
    source_metadata = {
        "source": "HRT_Protocol_Test.pdf",
        "data_source": "protocols",
        "content_type": "clinical_protocol",
        "protocol_type": "menopause"
    }
    
    try:
        chunks = vector_store.create_medical_chunks(sample_clinical_text, source_metadata)
        print(f"✅ Created {len(chunks)} medical chunks")
        
        # Display chunk information
        for i, chunk in enumerate(chunks):
            print(f"\n   Chunk {i+1}:")
            print(f"   📊 Size: {len(chunk.page_content)} characters")
            print(f"   🏥 Section: {chunk.metadata.get('clinical_section', 'unknown')}")
            print(f"   🔬 Concepts: {', '.join(chunk.metadata.get('medical_concepts', []))}")
            print(f"   ⚠️  Contraindications: {chunk.metadata.get('contains_contraindications', False)}")
            print(f"   💊 Dosing: {chunk.metadata.get('contains_dosing', False)}")
            print(f"   📋 Content preview: {chunk.page_content[:100]}...")
            
    except Exception as e:
        print(f"❌ Medical chunking failed: {e}")
        return
    
    # Test embedding generation
    print("\n3. Testing Medical Embedding Generation...")
    try:
        test_texts = [
            "Patient with breast cancer history contraindicated for hormone therapy",
            "Estradiol 1mg daily for moderate vasomotor symptoms",
            "Monitor for venous thromboembolism in HRT patients"
        ]
        
        embeddings = vector_store.embedding_model.encode(test_texts)
        print(f"✅ Generated embeddings with shape: {embeddings.shape}")
        print(f"   📐 Embedding dimensions: {embeddings.shape[1]}")
        print(f"   🧬 Model: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return
    
    # Test adding documents to collection
    print("\n4. Testing Document Addition to Medical Collection...")
    try:
        # Add the chunks to the vector store
        doc_ids = vector_store.add_documents(chunks)
        print(f"✅ Added {len(doc_ids)} documents to medical collection")
        
        # Get collection stats
        stats = vector_store.get_collection_stats()
        print(f"   📊 Total documents in collection: {stats.get('total_documents', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Document addition failed: {e}")
        return
    
    # Test medical search
    print("\n5. Testing Medical Domain Search...")
    try:
        test_queries = [
            "contraindications for hormone replacement therapy",
            "estradiol dosing for menopause symptoms",
            "monitoring requirements for HRT patients"
        ]
        
        for query in test_queries:
            print(f"\n   🔍 Query: '{query}'")
            results = vector_store.search_with_scores(query, k=2)
            
            for doc, score in results:
                print(f"   📄 Score: {score:.3f}")
                print(f"   🏥 Section: {doc.metadata.get('clinical_section', 'unknown')}")
                print(f"   📝 Content: {doc.page_content[:80]}...")
                
    except Exception as e:
        print(f"❌ Medical search failed: {e}")
        return
    
    print("\n" + "=" * 50)
    print("🎉 Medical Embeddings System Test Complete!")
    print("\n📈 Key Improvements:")
    print("   • Medical domain embeddings (PubMedBERT)")
    print("   • Clinical-aware text chunking")
    print("   • Medical concept extraction")
    print("   • Enhanced metadata for clinical search")
    print("   • Larger chunks for better clinical context")

if __name__ == "__main__":
    test_medical_embeddings()