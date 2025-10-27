import logging
import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")

# Add utils to path for VectorStoreManager
sys.path.append(str(Path(__file__).parent / 'utils'))

from VectorStoreManager import VectorStoreManager
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

# For LLM integration
try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import openai
except ImportError:
    openai = None

logger = logging.getLogger(__name__)

class RAGEngine:
    """Main RAG engine for querying clinical protocols."""
    
    def __init__(self, 
                 vector_store_path: str = "./data/chroma_db",
                 model_provider: str = "anthropic",
                 model_name: str = "claude-3-5-sonnet-20241022"):
        
        self.vector_store = VectorStoreManager(vector_store_path)
        self.model_provider = model_provider
        self.model_name = model_name
        
        # Initialize LLM client with environment variables or Streamlit secrets
        if model_provider == "anthropic" and anthropic:
            # Try to get API key from environment or Streamlit secrets
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                try:
                    import streamlit as st
                    api_key = st.secrets.get("ANTHROPIC_API_KEY")
                except:
                    pass
            
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
                logger.info(f"✅ Initialized Anthropic client with model: {model_name}")
            else:
                logger.warning("⚠️  ANTHROPIC_API_KEY not found in environment or secrets. Using fallback mode.")
                self.client = None
        elif model_provider == "openai" and openai:
            # Try to get API key from environment or Streamlit secrets
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                try:
                    import streamlit as st
                    api_key = st.secrets.get("OPENAI_API_KEY")
                except:
                    pass
            
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
                logger.info(f"✅ Initialized OpenAI client with model: {model_name}")
            else:
                logger.warning("⚠️  OPENAI_API_KEY not found in environment or secrets. Using fallback mode.")
                self.client = None
        else:
            logger.warning(f"⚠️  LLM provider {model_provider} not available. Using fallback mode.")
            self.client = None

    def _clean_text(self, text: str) -> str:
        """Clean extracted text from PDFs and other sources."""
        if not text:
            return ""
        
        # Remove extra spaces between words
        text = re.sub(r'\s+', ' ', text)
        
        # Remove strange characters and fix common PDF extraction issues
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'[^\w\s.,!?;:()\-\'"/$%&]', ' ', text)
        
        # Fix multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Strip and clean up
        text = text.strip()
        
        return text

    def _format_source_filename(self, source_path: str) -> str:
        """Format source filename for better display."""
        if not source_path:
            return "Unknown source"
        
        filename = Path(source_path).name
        
        # Clean up filename
        filename = filename.replace("Copy of ", "")
        filename = filename.replace("(published)", "")
        filename = filename.replace(".pdf", "")
        filename = filename.replace(".docx", "")
        filename = filename.replace("_", " ")
        
        # Clean extra spaces
        filename = re.sub(r' +', ' ', filename).strip()
        
        return filename

    def _enhance_query_for_comprehensive_search(self, question: str) -> List[str]:
        """Generate multiple search queries to capture comprehensive information."""
        
        base_queries = [question]
        
        # If asking about breast cancer and hormone therapy, search for multiple aspects
        if "breast cancer" in question.lower() and any(term in question.lower() for term in ["hormone", "hrt", "replacement"]):
            base_queries.extend([
                "hormone receptor positive breast cancer contraindications",
                "hormone receptor negative breast cancer hormone therapy",
                "breast cancer survivors hormone replacement therapy",
                "DCIS LCIS atypical hyperplasia hormone therapy",
                "tamoxifen aromatase inhibitors hormone therapy contraindications",
                "vaginal estrogen breast cancer patients",
                "shared decision making breast cancer hormone therapy"
            ])
        
        # Add other condition-specific expansions
        elif "glp-1" in question.lower() or "semaglutide" in question.lower():
            base_queries.extend([
                "GLP-1 eligibility criteria BMI requirements",
                "semaglutide monitoring side effects",
                "weight management medication contraindications"
            ])
        
        elif "sleep" in question.lower() and "menopause" in question.lower():
            base_queries.extend([
                "sleep hygiene perimenopausal patients",
                "melatonin hormone changes sleep disturbances",
                "non-hormonal sleep interventions menopause"
            ])
        
        return base_queries

    def _comprehensive_search(self, question: str, k: int = 15, protocol_type: Optional[str] = None) -> List[tuple]:
        """Perform comprehensive search using multiple strategies with source prioritization."""
        
        search_filter = {"protocol_type": protocol_type} if protocol_type else None
        all_results = {}
        
        # Get enhanced queries
        search_queries = self._enhance_query_for_comprehensive_search(question)
        
        # Search with all queries
        for i, query in enumerate(search_queries):
            try:
                results = self.vector_store.search_with_scores(
                    query=query,
                    k=k,
                    filter_dict=search_filter
                )
                
                # Weight results - original query gets highest weight
                weight = 1.0 if i == 0 else 0.7
                
                for doc, score in results:
                    # Apply source type prioritization boost
                    data_source = doc.metadata.get("data_source", "protocols")
                    source_boost = self._get_source_priority_boost(data_source)
                    
                    # Apply the boost to the score
                    boosted_score = score * weight * source_boost
                    
                    doc_id = str(hash(doc.page_content[:100]))
                    if doc_id in all_results:
                        # Boost score for documents found in multiple searches
                        existing_score = all_results[doc_id][1]
                        all_results[doc_id] = (doc, max(existing_score, boosted_score) + 0.1)
                    else:
                        all_results[doc_id] = (doc, boosted_score)
                        
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {str(e)}")
                continue
        
        # Sort by combined score and return top results
        combined_results = list(all_results.values())
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        return combined_results[:k]

    def _get_source_priority_boost(self, data_source: str) -> float:
        """Get priority boost multiplier based on source type for clinical queries."""
        source_priorities = {
            "protocols": 1.5,           # Highest priority for clinical protocols
            "midi_zendesk_articles": 1.2,  # Medium priority for support articles
            "midi_blog_posts": 1.0      # Standard priority for blog posts
        }
        return source_priorities.get(data_source, 1.0)

    def _extract_relevant_section(self, content: str, max_length: int = 800) -> str:
        """Extract relevant section starting from logical beginning points."""
        if not content or len(content) <= max_length:
            return content
        
        # Clean the content first
        cleaned = self._clean_text(content)
        
        # Look for natural section breaks near the beginning
        section_markers = [
            '. ', '.\n', '• ', '- ', 
            'BACKGROUND:', 'OVERVIEW:', 'SUMMARY:', 'INDICATION:', 'CRITERIA:', 
            'CLINICAL GUIDANCE:', 'PROTOCOL:', 'RECOMMENDATION:', 'CONTRAINDICATION:',
            'Patient should', 'Patients with', 'For patients', 'When considering'
        ]
        
        # Find the best starting point (prefer beginning of sentences/sections)
        best_start = 0
        for marker in section_markers:
            marker_pos = cleaned.find(marker)
            if 0 <= marker_pos <= 100:  # Within first 100 characters
                if marker.endswith(':'):
                    best_start = marker_pos
                    break
                elif marker in ['. ', '.\n'] and marker_pos > 0:
                    best_start = marker_pos + len(marker)
                    break
        
        # Extract from best starting point
        section_text = cleaned[best_start:]
        
        # If still too long, find a good ending point (complete sentences)
        if len(section_text) > max_length:
            # Look for sentence endings near the max length
            cutoff = max_length
            for i in range(max_length - 50, min(len(section_text), max_length + 100)):
                if section_text[i:i+2] in ['. ', '.\n']:
                    cutoff = i + 1
                    break
            
            section_text = section_text[:cutoff]
            if cutoff < len(cleaned[best_start:]):
                section_text += "..."
        
        return section_text.strip()

    def query(self, 
              question: str, 
              patient_context: Optional[Dict[str, Any]] = None,
              k: int = 10,
              protocol_type: Optional[str] = None) -> Dict[str, Any]:
        """Query the RAG system with a clinical question."""
        try:
            # Use comprehensive search for better coverage
            relevant_docs = self._comprehensive_search(question, k=k, protocol_type=protocol_type)
            
            if not relevant_docs:
                return {
                    "answer": "**INSUFFICIENT DATA**\nNo relevant protocol information found for this query.\n\n**RECOMMENDATION:**\nConsult primary clinical guidelines or seek specialist input.",
                    "sources": [],
                    "confidence": 0.0,
                    "query": question,
                    "patient_context": patient_context
                }
            
            # Generate comprehensive answer
            answer = self._generate_comprehensive_answer(question, relevant_docs, patient_context)
            
            # Prepare sources with cleaned formatting - show more text and all 5 sources
            sources = []
            for doc, score in relevant_docs[:5]:  # Ensure we get exactly 5 sources
                # Clean the content text
                cleaned_content = self._clean_text(doc.page_content)
                
                # Format filename nicely
                formatted_filename = self._format_source_filename(doc.metadata.get("source", "Unknown"))
                
                # Get data source information
                data_source = doc.metadata.get("data_source", "protocols")
                content_type = doc.metadata.get("content_type", "clinical_protocol")
                
                # Format source display with data source info
                source_display = self._format_source_display(formatted_filename, data_source, content_type)
                
                # Extract relevant section starting from logical beginning
                content_preview = self._extract_relevant_section(cleaned_content, max_length=800)
                
                sources.append({
                    "content": content_preview,
                    "source": source_display,
                    "data_source": data_source,
                    "content_type": content_type,
                    "protocol_type": doc.metadata.get("protocol_type", "general"),
                    "confidence_score": float(score)
                })
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": max([score for _, score in relevant_docs]) if relevant_docs else 0.0,
                "query": question,
                "patient_context": patient_context
            }
            
        except Exception as e:
            logger.error(f"Error in RAG query: {str(e)}")
            return {
                "answer": f"An error occurred while processing your query: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "query": question,
                "patient_context": patient_context
            }

    def _generate_comprehensive_answer(self, 
                                     question: str, 
                                     relevant_docs: List[tuple], 
                                     patient_context: Optional[Dict[str, Any]] = None) -> str:
        """Generate comprehensive clinical answers with nuanced guidance."""
        
        if not self.client:
            return self._fallback_comprehensive_answer(question, relevant_docs)
        
        # Build comprehensive context from more documents with cleaned text
        cleaned_context_parts = []
        for i, (doc, score) in enumerate(relevant_docs[:6]):
            cleaned_content = self._clean_text(doc.page_content)
            cleaned_context_parts.append(f"Protocol Source {i+1}:\n{cleaned_content}")
        
        context = "\n\n".join(cleaned_context_parts)
        
        # Build patient context string
        patient_info = ""
        if patient_context:
            patient_info = f"\nPatient Context: {', '.join([f'{k}: {v}' for k, v in patient_context.items()])}"
        
        # Create comprehensive clinical prompt
        prompt = f"""You are an expert clinical decision support AI for Midi Health protocols. Provide comprehensive, nuanced clinical guidance.

Clinical Protocol Context:
{context}

Question: {question}{patient_info}

INSTRUCTIONS FOR COMPREHENSIVE CLINICAL RESPONSE:

1. PROVIDE DETAILED, CATEGORIZED GUIDANCE:
   - Address different clinical scenarios and subtypes when relevant
   - Include specific conditions, contraindications, and considerations
   - Mention timing considerations (e.g., years post-diagnosis)
   - Address both systemic and local treatment options when applicable

2. STRUCTURE YOUR RESPONSE WITH CLEAR SECTIONS:
   - Use descriptive subheadings for different clinical scenarios
   - Provide specific guidance for each category
   - Include shared decision-making considerations
   - Mention consultation requirements with specialists when appropriate

3. BE COMPREHENSIVE BUT PRACTICAL:
   - Cover all relevant clinical scenarios mentioned in the protocols
   - Include specific medication names, formulations, and preferences when mentioned
   - Address contraindications and special considerations
   - Provide clear guidance on when exceptions might be considered

4. MAINTAIN CLINICAL ACCURACY:
   - Base all recommendations strictly on the provided protocol information
   - If protocols don't cover a scenario, clearly state this limitation
   - Use appropriate medical terminology while remaining clear
   - Include relevant risk factors and monitoring requirements

5. FORMAT FOR CLINICAL USE:
   - Use clear subheadings and bullet points
   - Make recommendations actionable for clinicians
   - Include specific timeframes and criteria when mentioned in protocols
   - End with general contraindications or key takeaways when appropriate

Provide a comprehensive clinical response:"""

        try:
            if self.model_provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=1500,  # Increased for comprehensive answers
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            
            elif self.model_provider == "openai":
                openai_model = "gpt-4" if "gpt-4" in self.model_name else "gpt-3.5-turbo"
                response = self.client.chat.completions.create(
                    model=openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1500
                )
                return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return self._fallback_comprehensive_answer(question, relevant_docs)

    def _fallback_comprehensive_answer(self, question: str, relevant_docs: List[tuple]) -> str:
        """Provide comprehensive fallback answers when LLM is not available."""
        if not relevant_docs:
            return "**INSUFFICIENT DATA**\nNo relevant protocol information found for this query.\n\n**RECOMMENDATION:**\nConsult primary clinical guidelines or seek specialist input."
        
        # Organize content by protocol type and relevance
        organized_content = {}
        
        for doc, score in relevant_docs[:5]:
            protocol_type = doc.metadata.get('protocol_type', 'general')
            if protocol_type not in organized_content:
                organized_content[protocol_type] = []
            
            # Clean the content before organizing
            cleaned_content = self._clean_text(doc.page_content)
            organized_content[protocol_type].append((cleaned_content, score))
        
        # Build comprehensive fallback response
        answer_parts = [f"**CLINICAL GUIDANCE BASED ON PROTOCOLS**\n"]
        
        for protocol_type, contents in organized_content.items():
            answer_parts.append(f"\n**{protocol_type.replace('_', ' ').title()} Protocol Guidelines:**")
            
            for i, (content, score) in enumerate(contents[:2]):
                preview = content[:400] + "..." if len(content) > 400 else content
                answer_parts.append(f"\n• {preview}")
        
        answer_parts.append(f"\n\n**CLINICAL RECOMMENDATION:**")
        answer_parts.append("Review complete protocol sections and consider individual patient factors. Consult specialists when protocols indicate shared decision-making or complex medical situations.")
        
        return "\n".join(answer_parts)
    
    def _format_source_display(self, filename: str, data_source: str, content_type: str) -> str:
        """Format source display with data source information."""
        source_type_labels = {
            "protocols": "📋 Clinical Protocol",
            "midi_blog_posts": "📰 Midi Blog",
            "midi_zendesk_articles": "❓ Support Article"
        }
        
        source_label = source_type_labels.get(data_source, "📄 Document")
        return f"{source_label}: {filename}"
    
    def get_protocol_types(self) -> List[str]:
        """Get available protocol types."""
        return ["menopause", "weight_management", "diabetes", "thyroid", "sleep", "general"]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        vector_stats = self.vector_store.get_collection_stats()
        return {
            "vector_store": vector_stats,
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "available_protocol_types": self.get_protocol_types(),
            "llm_available": self.client is not None
        }