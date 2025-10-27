"""
Midi RAG Web Interface
A Streamlit web application providing both patient and clinical RAG interfaces
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from patient_rag_engine import PatientRAGEngine
from rag_engine import RAGEngine

# Page configuration
st.set_page_config(
    page_title="Midi RAG System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .interface-header {
        font-size: 1.8rem;
        color: #2e7d32;
        border-bottom: 2px solid #2e7d32;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .chat-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .assistant-message {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1976d2;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #f3e5f5;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #7b1fa2;
        margin: 1rem 0;
        text-align: left;
    }
    .confidence-score {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff8e1;
        border: 1px solid #ffb74d;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def check_password(interface_type):
    """Check password for the given interface type"""
    if interface_type == "patient":
        correct_password = st.secrets.get("PATIENT_PASSWORD", "midi-patient-2025")
        session_key = "patient_authenticated"
        title = "Patient Interface Access"
    else:
        correct_password = st.secrets.get("CLINICAL_PASSWORD", "midi-clinical-2025")
        session_key = "clinical_authenticated"
        title = "Clinical Interface Access"
    
    # Check if already authenticated
    if st.session_state.get(session_key, False):
        return True
    
    # Show password input
    st.markdown(f"### 🔒 {title}")
    st.markdown("Please enter the password to access this interface:")
    
    password = st.text_input("Password", type="password", key=f"{interface_type}_password")
    
    if st.button(f"Access {interface_type.title()} Interface", key=f"{interface_type}_login"):
        if password == correct_password:
            st.session_state[session_key] = True
            st.success(f"✅ Access granted to {interface_type} interface!")
            st.rerun()
        else:
            st.error("❌ Invalid password. Please try again.")
    
    return False

@st.cache_resource
def load_patient_rag():
    """Load and cache the patient RAG engine"""
    try:
        engine = PatientRAGEngine()
        return engine
    except Exception as e:
        st.error(f"Error loading Patient RAG engine: {str(e)}")
        return None

@st.cache_resource
def load_clinical_rag():
    """Load and cache the clinical RAG engine"""
    try:
        engine = RAGEngine()
        return engine
    except Exception as e:
        st.error(f"Error loading Clinical RAG engine: {str(e)}")
        return None

def patient_interface():
    """Patient-facing RAG interface"""
    st.markdown('<div class="interface-header">💬 MidiChat - Your Personal Health Companion</div>', unsafe_allow_html=True)
    
    # Load patient RAG engine
    patient_rag = load_patient_rag()
    if not patient_rag:
        st.error("Unable to load patient interface. Please try again later.")
        return
    
    # Display connection status
    try:
        doc_count = patient_rag.vector_store.collection.count()
        st.success(f"✅ Ready to chat! (Connected to {doc_count} documents)")
    except Exception as e:
        st.success("✅ Ready to chat! (Vector store connected)")
    
    # Chat interface
    st.markdown("---")
    st.markdown("### Ask your health question:")
    
    # Initialize chat history
    if "patient_messages" not in st.session_state:
        st.session_state.patient_messages = []
    
    # Chat input using form (better for clearing)
    with st.form("patient_question_form", clear_on_submit=True):
        user_question = st.text_area(
            "Enter your question for Midibot here:",
            height=100,
            placeholder="Feel free to share what's on your mind - whether it's symptoms, concerns, questions about treatment, or anything else."
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submitted = st.form_submit_button("Send Question", type="primary")
        with col2:
            if st.form_submit_button("Clear Chat"):
                st.session_state.patient_messages = []
                st.rerun()
    
    # Process the question if form was submitted
    if submitted and user_question.strip():
        # Add user message to history
        st.session_state.patient_messages.append({"role": "user", "content": user_question})
        
        # Get response from patient RAG
        with st.spinner("Thinking..."):
            try:
                result = patient_rag.chat(user_question)
                # Extract just the response text from the result dictionary
                if isinstance(result, dict) and 'response' in result:
                    response = result['response']
                else:
                    response = str(result)
                st.session_state.patient_messages.append({"role": "assistant", "content": response})
                st.rerun()
            except Exception as e:
                st.error(f"Error getting response: {str(e)}")
    
    # Display chat history (most recent first, but question before answer within each pair)
    if st.session_state.patient_messages:
        st.markdown("### Chat History:")
        
        # Group messages into Q&A pairs and reverse the pairs order
        messages = st.session_state.patient_messages
        qa_pairs = []
        
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                # We have both question and answer
                qa_pairs.append((messages[i], messages[i + 1]))
            else:
                # We have just a question (waiting for answer)
                qa_pairs.append((messages[i], None))
        
        # Display pairs in reverse order (most recent first)
        for question_msg, answer_msg in reversed(qa_pairs):
            # Display question first
            if question_msg["role"] == "user":
                st.markdown(f'<div class="user-message"><strong>You:</strong> {question_msg["content"]}</div>', unsafe_allow_html=True)
            
            # Then display answer (if it exists)
            if answer_msg and answer_msg["role"] == "assistant":
                st.markdown(f'<div class="assistant-message"><strong>MidiBot:</strong> {answer_msg["content"]}</div>', unsafe_allow_html=True)

def clinical_interface():
    """Clinical-facing RAG interface"""
    st.markdown('<div class="interface-header">🏥 Midi Clinical Protocol RAG System</div>', unsafe_allow_html=True)
    
    # Load clinical RAG engine
    clinical_rag = load_clinical_rag()
    if not clinical_rag:
        st.error("Unable to load clinical interface. Please try again later.")
        return
    
    # Display system info
    col1, col2, col3 = st.columns(3)
    with col1:
        try:
            doc_count = clinical_rag.vector_store.collection.count()
            st.metric("📊 Documents", f"{doc_count}")
        except Exception:
            st.metric("📊 Documents", "Connected")
    with col2:
        st.metric("🧠 Model", "Claude 3.5 Sonnet")
    with col3:
        st.metric("🔑 LLM Status", "✅ Available" if clinical_rag.client else "❌ Unavailable")
    
    # Available protocol types
    protocol_types = ["menopause", "weight_management", "diabetes", "thyroid", "sleep", "general"]
    st.info(f"📋 Available Protocol Types: {', '.join(protocol_types)}")
    
    # Clinical query interface
    st.markdown("---")
    st.markdown("### Clinical Question:")
    
    # Initialize clinical chat history
    if "clinical_messages" not in st.session_state:
        st.session_state.clinical_messages = []
    
    # Clinical input using form (better for clearing)
    with st.form("clinical_question_form", clear_on_submit=True):
        clinical_question = st.text_area(
            "Enter your clinical question:",
            height=100,
            placeholder="Ask about clinical protocols, treatment guidelines, contraindications, etc."
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submitted = st.form_submit_button("Search Protocols", type="primary")
        with col2:
            if st.form_submit_button("Clear History"):
                st.session_state.clinical_messages = []
                st.rerun()
    
    # Process the question if form was submitted
    if submitted and clinical_question.strip():
        # Add user message to history
        st.session_state.clinical_messages.append({"role": "user", "content": clinical_question})
        
        # Get response from clinical RAG
        with st.spinner("🔍 Searching protocols..."):
            try:
                result = clinical_rag.query(clinical_question)
                
                # Display confidence score
                confidence = result.get("confidence", 0)
                confidence_color = "#4caf50" if confidence > 0.7 else "#ff9800" if confidence > 0.4 else "#f44336"
                
                response_data = {
                    "role": "assistant",
                    "content": result["answer"],
                    "confidence": confidence,
                    "sources": result.get("sources", [])
                }
                st.session_state.clinical_messages.append(response_data)
                
            except Exception as e:
                st.error(f"Error getting response: {str(e)}")
    
    # Display clinical chat history (most recent first, but question before answer within each pair)
    if st.session_state.clinical_messages:
        st.markdown("### Clinical Responses:")
        
        # Group messages into Q&A pairs and reverse the pairs order
        messages = st.session_state.clinical_messages
        qa_pairs = []
        
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                # We have both question and answer
                qa_pairs.append((messages[i], messages[i + 1]))
            else:
                # We have just a question (waiting for answer)
                qa_pairs.append((messages[i], None))
        
        # Display pairs in reverse order (most recent first)
        for question_msg, answer_msg in reversed(qa_pairs):
            # Display question first
            if question_msg["role"] == "user":
                st.markdown(f'<div class="user-message"><strong>Clinical Question:</strong> {question_msg["content"]}</div>', unsafe_allow_html=True)
            
            # Then display answer (if it exists)
            if answer_msg and answer_msg["role"] == "assistant":
                # Show confidence score
                confidence = answer_msg.get("confidence", 0)
                confidence_color = "#4caf50" if confidence > 0.7 else "#ff9800" if confidence > 0.4 else "#f44336"
                st.markdown(f'<div class="confidence-score">📋 <strong>Confidence Score:</strong> <span style="color: {confidence_color};">{confidence:.2f}</span></div>', unsafe_allow_html=True)
                
                # Show answer
                st.markdown(f'<div class="assistant-message">{answer_msg["content"]}</div>', unsafe_allow_html=True)
                
                # Show sources if available
                sources = answer_msg.get("sources", [])
                if sources:
                    with st.expander(f"📚 View Sources ({len(sources)} found)"):
                        for i, source in enumerate(sources, 1):
                            # Format like CLI: filename (protocol_type) - Score: confidence
                            source_title = source.get('source', 'Unknown')
                            protocol_type = source.get('protocol_type', 'general')
                            confidence = source.get('confidence_score', 0)
                            data_source = source.get('data_source', 'unknown')
                            
                            # Clean up source title for better display
                            if source_title.startswith('📋 Clinical Protocol: '):
                                display_title = source_title.replace('📋 Clinical Protocol: ', '')
                            elif source_title.startswith('📰 Midi Blog: '):
                                display_title = source_title.replace('📰 Midi Blog: ', '')
                            elif source_title.startswith('❓ Support Article: '):
                                display_title = source_title.replace('❓ Support Article: ', '')
                            else:
                                display_title = source_title
                            
                            # Format header like CLI
                            st.markdown(f"**{i}. {display_title}** ({protocol_type}) - Score: {confidence:.3f}")
                            st.markdown(f"*Data Source: {data_source}*")
                            
                            # Show full content like CLI (already truncated by RAG engine)
                            if source.get("content"):
                                content = source["content"]
                                # Use code block for better formatting and readability
                                st.text_area(
                                    f"Content from source {i}:",
                                    content,
                                    height=150,
                                    key=f"source_content_{i}_{hash(content[:50])}",
                                    disabled=True
                                )
                            
                            if i < len(sources):  # Add separator between sources except for last one
                                st.markdown("---")

def main():
    """Main application"""
    # Header
    st.markdown('<div class="main-header">🏥 Midi RAG System</div>', unsafe_allow_html=True)
    
    # Sidebar for interface selection
    st.sidebar.title("🔧 Interface Selection")
    interface_choice = st.sidebar.radio(
        "Choose Interface:",
        ["Patient Interface", "Clinical Interface"],
        help="Select the appropriate interface for your role"
    )
    
    # Add logout buttons
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout Patient"):
        st.session_state.patient_authenticated = False
        st.session_state.patient_messages = []
    
    if st.sidebar.button("🚪 Logout Clinical"):
        st.session_state.clinical_authenticated = False
        st.session_state.clinical_messages = []
    
    # Information section
    with st.sidebar.expander("ℹ️ About This System"):
        st.markdown("""
        **Midi RAG System** provides two specialized interfaces:
        
        **Patient Interface:**
        - Conversational health support
        - Educational information
        - Treatment guidance
        
        **Clinical Interface:**
        - Protocol searches
        - Evidence-based guidelines
        - Confidence scoring
        - Source documentation
        """)
    
    # Route to appropriate interface
    if interface_choice == "Patient Interface":
        if check_password("patient"):
            patient_interface()
    else:
        if check_password("clinical"):
            clinical_interface()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
        "Midi RAG System - Powered by Claude 3.5 Sonnet & ChromaDB"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()