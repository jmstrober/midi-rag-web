#!/usr/bin/env python3
"""
Medical RAG Streamlit App
Uses the new medical embeddings system for clinical queries
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set environment variables early to prevent multiprocessing issues
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

# DO NOT configure PyTorch threading - this causes segmentation faults!

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "src"))

from rag_engine import RAGEngine
from patient_rag_engine import PatientRAGEngine

# Configure page
st.set_page_config(
    page_title="MIDI Medical RAG System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better readability
st.markdown("""
<style>
.source-container {
    background-color: #f8f9fa;
    border-left: 4px solid #007bff;
    padding: 15px;
    margin: 10px 0;
    border-radius: 0 8px 8px 0;
}

.source-title {
    font-weight: bold;
    color: #007bff;
    margin-bottom: 8px;
    font-size: 1.1em;
}

.source-content {
    color: #495057;
    line-height: 1.6;
    background-color: white;
    padding: 12px;
    border-radius: 6px;
    border: 1px solid #e9ecef;
}

.medical-badge {
    background-color: #28a745;
    color: white;
    padding: 4px 8px;
    border-radius: 12px;
    font-size: 0.8em;
    margin-right: 8px;
}

.test-results {
    background-color: #e7f3ff;
    border: 1px solid #b3d9ff;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_engines():
    """Initialize both RAG engines with enhanced error handling."""
    try:
        # Import torch but don't configure threading to avoid conflicts
        import torch
        
        clinical_engine = RAGEngine()
        
        patient_engine = PatientRAGEngine()
        
        # Check if using fallback mode
        if hasattr(clinical_engine.vector_store, 'use_fallback') and clinical_engine.vector_store.use_fallback:
            st.warning("ℹ️ **Using text-based search mode** - The app is running with simplified search instead of AI embeddings for better stability.")
        else:
            st.success("✅ **Using AI embeddings** - Full vector search enabled with medical knowledge base.")
        
        return clinical_engine, patient_engine
    except Exception as e:
        error_msg = f"Failed to initialize RAG engines: {str(e)}"
        st.error(error_msg)
        st.error("**Troubleshooting suggestions:**")
        st.error("1. The app may be loading - please wait and refresh")
        st.error("2. Try clearing browser cache and refreshing")
        st.error("3. Check Streamlit Community Cloud logs for detailed errors")
        
        # Log the full error for debugging
        import traceback
        st.code(traceback.format_exc())
        
        return None, None

def display_sources_with_scores(sources, interface_type="patient"):
    """Display sources with improved formatting."""
    if not sources:
        st.warning("No sources found.")
        return
    
    # Filter to show only protocols for clinical interface
    if interface_type == "clinical":
        sources = [s for s in sources if 'protocol' in s.get('source', '').lower()]
        if not sources:
            st.warning("No protocol sources found for this query.")
            return
    
    st.markdown("### 📚 Sources")
    
    for i, source in enumerate(sources, 1):
        # Extract metadata
        source_name = source.get('source', f'Source {i}')
        content = source.get('content', 'No content available')
        section = source.get('metadata', {}).get('clinical_section', 'general')
        concepts = source.get('metadata', {}).get('medical_concepts', [])
        
        # Create source container
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"""
                <div class="source-container">
                    <div class="source-title">
                        🏥 {source_name}
                    </div>
                    <div style="margin-bottom: 8px;">
                        <span class="medical-badge">Section: {section}</span>
                    </div>
                    <div class="source-content">
                        {content}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if concepts:
                    st.markdown("**Medical Concepts:**")
                    for concept in concepts[:5]:  # Show first 5 concepts
                        st.markdown(f"• {concept}")

def main():
    """Main Streamlit app."""
    
    st.title("🏥 MIDI Medical RAG System")
    
    # Initialize session state for authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'interface_type' not in st.session_state:
        st.session_state.interface_type = "Patient Interface"
    if 'current_authenticated_interface' not in st.session_state:
        st.session_state.current_authenticated_interface = None
    
    # Initialize conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # Sidebar for interface selection
    st.sidebar.title("🏥 MIDI Medical RAG")
    st.sidebar.markdown("### Interface Selection")
    
    # Interface selection with radio buttons
    selected_interface = st.sidebar.radio(
        "Choose Interface:",
        ["Patient Interface", "Clinical Interface"],
        index=0 if st.session_state.interface_type == "Patient Interface" else 1
    )
    
    # Check if interface has changed
    if selected_interface != st.session_state.interface_type:
        st.session_state.interface_type = selected_interface
        st.session_state.authenticated = False
        st.session_state.current_authenticated_interface = None
        # Clear conversation when switching interfaces
        st.session_state.conversation_history = []
        st.rerun()
    
    # Initialize engines
    clinical_engine, patient_engine = initialize_engines()
    
    # Authentication logic
    if not st.session_state.authenticated or st.session_state.current_authenticated_interface != selected_interface:
        st.markdown("---")
        
        if selected_interface == "Patient Interface":
            st.markdown("### 💬 Patient Interface")
            st.markdown("*Patient-friendly explanations and guidance*")
            required_password = "midi-patient-2025"
            
        else:  # Clinical Interface
            st.markdown("### 🩺 Clinical Interface")
            st.markdown("*Specialized medical queries with protocol prioritization*")
            required_password = "midi-clinical-2025"
        
        # Password input
        password = st.text_input(
            f"Enter {selected_interface.lower()} password:",
            type="password",
            key=f"password_{selected_interface}"
        )
        
        if password:
            if password == required_password:
                st.session_state.authenticated = True
                st.session_state.current_authenticated_interface = selected_interface
                st.success(f"✅ Access granted to {selected_interface.lower()}")
                st.rerun()
            else:
                st.error("❌ Incorrect password. Please try again.")
        else:
            st.info(f"Please enter the password to access the {selected_interface.lower()}.")
        
        return
    
    # Main interface (authenticated)
    if selected_interface == "Clinical Interface":
        if not clinical_engine:
            st.error("Clinical engine not available")
            return
        current_engine = clinical_engine
        placeholder_text = "Enter your clinical question (e.g., 'What are contraindications for HRT in breast cancer survivors?')"
        
    else:  # Patient Interface
        if not patient_engine:
            st.error("Patient engine not available")
            return
        current_engine = patient_engine
        placeholder_text = "Ask your health question (e.g., 'What symptoms might I experience during menopause?')"
    
    # Conversation interface
    st.markdown("---")
    
    # Create two columns for layout - question input and reset button at top
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### ❓ Ask a Question")
    
    with col2:
        if st.button("🔄 Reset", use_container_width=True, help="Clear conversation and start fresh"):
            st.session_state.conversation_history = []
            # Clear conversation history and let the UI update naturally
    
    # Determine if this is a follow-up
    is_followup = len(st.session_state.conversation_history) > 0
    
    if is_followup:
        st.info("💡 You can ask follow-up questions based on our previous conversation, or use Reset to start fresh.")
    
    # Query input form - ALWAYS at the top
    with st.form("query_form", clear_on_submit=True):
        query = st.text_area(
            "Your Question:" if not is_followup else "Your Follow-up Question:",
            placeholder=placeholder_text if not is_followup else "Ask a follow-up question about the previous topic...",
            height=100
        )
        
        submit = st.form_submit_button("🔍 Search", use_container_width=True)
    
    # Process query
    if submit and query.strip():
        with st.spinner("🔍 Searching medical knowledge base..."):
            try:
                # Build context-aware query for follow-ups
                if is_followup:
                    # Get last 2 Q&As for context
                    recent_context = st.session_state.conversation_history[-2:]
                    context_summary = []
                    
                    for conv in recent_context:
                        context_summary.append(f"Previous Q: {conv['question']}")
                        context_summary.append(f"Previous A: {conv['answer'][:200]}...")
                    
                    enhanced_query = f"Follow-up question context:\n{chr(10).join(context_summary)}\n\nCurrent question: {query}"
                else:
                    enhanced_query = query
                
                response = current_engine.query(enhanced_query)
                
                # Store conversation
                conversation_entry = {
                    "question": query,
                    "answer": response['answer'],
                    "sources": response.get('sources', []),
                    "timestamp": datetime.now().isoformat()
                }
                
                st.session_state.conversation_history.append(conversation_entry)
                
                # Keep only last 5 conversations for performance
                if len(st.session_state.conversation_history) > 5:
                    st.session_state.conversation_history = st.session_state.conversation_history[-5:]
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.info("Please try rephrasing your question or contact support.")
    
    # Display the most recent answer prominently (if any conversation exists)
    if st.session_state.conversation_history:
        most_recent = st.session_state.conversation_history[-1]
        st.markdown("---")
        st.markdown("## 💡 Your Answer")
        st.markdown(f"**Question:** {most_recent['question']}")
        st.markdown(most_recent['answer'])
        
        # Display sources for current answer - only for clinical interface
        if most_recent.get('sources') and selected_interface == "Clinical Interface":
            interface_for_sources = "clinical"
            display_sources_with_scores(most_recent['sources'], interface_for_sources)
    
    # Display previous conversations (excluding the most recent one which is shown above)
    if len(st.session_state.conversation_history) > 1:
        st.markdown("---")
        st.markdown("### 📜 Previous Questions")
        
        # Show previous conversations (all except the most recent one)
        previous_conversations = st.session_state.conversation_history[:-1]  # Exclude the last one
        
        for i, conversation in enumerate(reversed(previous_conversations)):
            question_num = len(previous_conversations) - i
            
            with st.expander(f"Q{question_num}: {conversation['question'][:80]}...", expanded=False):
                st.markdown(f"**Question:** {conversation['question']}")
                st.markdown(f"**Answer:** {conversation['answer']}")
                
                # Only show sources for clinical interface
                if conversation.get('sources') and selected_interface == "Clinical Interface":
                    interface_for_sources = "clinical"
                    display_sources_with_scores(conversation['sources'], interface_for_sources)
    
    # Sidebar information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Current Interface")
    st.sidebar.info(f"✅ {selected_interface}")
    
    if selected_interface == "Clinical Interface":
        st.sidebar.markdown("### 🎯 Clinical Features")
        st.sidebar.markdown(
            "• Protocol prioritization\n"
            "• Medical terminology\n"
            "• Clinical context\n"
            "• Contraindication detection\n"
            "• Dosing guidance"
        )

if __name__ == "__main__":
    main()