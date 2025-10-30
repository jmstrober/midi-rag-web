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
    try:
        from langchain.schema import Document
    except ImportError:
        from langchain_community.schema import Document

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
                 model_name: str = "claude-3-5-haiku-20241022"):
        
        self.vector_store = VectorStoreManager(vector_store_path)
        self.model_provider = model_provider
        self.model_name = model_name
        
        # Initialize LLM client with environment variables
        if model_provider == "anthropic" and anthropic:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
                logger.info(f"✅ Initialized Anthropic client with model: {model_name}")
            else:
                logger.warning("⚠️  ANTHROPIC_API_KEY not found in environment. Using fallback mode.")
                self.client = None
        elif model_provider == "openai" and openai:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
                logger.info(f"✅ Initialized OpenAI client with model: {model_name}")
            else:
                logger.warning("⚠️  OPENAI_API_KEY not found in environment. Using fallback mode.")
                self.client = None
        else:
            logger.warning(f"⚠️  LLM provider {model_provider} not available. Using fallback mode.")
            self.client = None

    def _extract_smart_excerpt(self, content: str, max_length: int = 600) -> str:
        """Extract a smart excerpt that preserves sentence boundaries and logical flow."""
        if not content or len(content) <= max_length:
            return content
        
        # Clean up the content first
        cleaned_content = self._clean_text(content)
        
        if len(cleaned_content) <= max_length:
            return cleaned_content
        
        # Find sentence boundaries using multiple delimiters
        sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        
        # Find the best cutoff point - try to get close to max_length but end at sentence boundary
        best_cutoff = 0
        
        # Look for sentence endings near the target length
        search_start = max(0, max_length - 150)  # Start looking 150 chars before target
        search_end = min(len(cleaned_content), max_length + 100)  # End 100 chars after target
        
        for i in range(search_start, search_end):
            for ending in sentence_endings:
                if cleaned_content[i:i+len(ending)] == ending:
                    # Found a sentence ending, check if it's a good stopping point
                    next_pos = i + len(ending)
                    if next_pos <= max_length + 50:  # Allow some buffer
                        best_cutoff = next_pos
                    elif best_cutoff == 0:  # No good cutoff found yet, use this one
                        best_cutoff = next_pos
        
        # If no good sentence boundary found, look for paragraph breaks
        if best_cutoff == 0:
            for i in range(search_start, search_end):
                if cleaned_content[i:i+2] == '\n\n' or cleaned_content[i:i+2] == '  ':
                    next_pos = i + 2
                    if next_pos <= max_length + 50:
                        best_cutoff = next_pos
                        break
        
        # If still no good cutoff, look for any period followed by space
        if best_cutoff == 0:
            last_period = cleaned_content.rfind('. ', 0, max_length + 50)
            if last_period > max_length - 200:  # Only use if reasonably close to target
                best_cutoff = last_period + 2
        
        # Fall back to original approach if nothing found
        if best_cutoff == 0 or best_cutoff < max_length // 2:
            return cleaned_content[:max_length] + "..."
        
        excerpt = cleaned_content[:best_cutoff].strip()
        
        # Add ellipsis if we cut off the content
        if best_cutoff < len(cleaned_content):
            # Don't add ellipsis if we ended with a sentence
            if not any(excerpt.endswith(ending.strip()) for ending in sentence_endings):
                excerpt += "..."
        
        return excerpt

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
        """Perform comprehensive search using multiple strategies."""
        
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
                    doc_id = str(hash(doc.page_content[:100]))
                    if doc_id in all_results:
                        # Boost score for documents found in multiple searches
                        existing_score = all_results[doc_id][1]
                        all_results[doc_id] = (doc, max(existing_score, score * weight) + 0.1)
                    else:
                        all_results[doc_id] = (doc, score * weight)
                        
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {str(e)}")
                continue
        
        # Sort by combined score and return top results
        combined_results = list(all_results.values())
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        return combined_results[:k]

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
                
                # Use smart excerpt extraction instead of simple truncation
                content_excerpt = self._extract_smart_excerpt(cleaned_content, max_length=600)
                
                # Format filename nicely
                formatted_filename = self._format_source_filename(doc.metadata.get("source", "Unknown"))
                
                # Get data source information
                data_source = doc.metadata.get("data_source", "protocols")
                content_type = doc.metadata.get("content_type", "clinical_protocol")
                
                # Format source display with data source info
                source_display = self._format_source_display(formatted_filename, data_source, content_type)
                
                sources.append({
                    "content": content_excerpt,
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
                # Use smart excerpt instead of simple truncation
                excerpt = self._extract_smart_excerpt(content, max_length=500)
                answer_parts.append(f"\n• {excerpt}")
        
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