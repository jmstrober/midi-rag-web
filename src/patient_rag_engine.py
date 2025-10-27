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

class PatientRAGEngine:
    """Patient-focused RAG engine for empathetic, conversational guidance."""
    
    def __init__(self, 
                 vector_store_path: str = "./data/chroma_db",
                 model_provider: str = "anthropic",
                 model_name: str = "claude-sonnet-4"):
        
        self.vector_store = VectorStoreManager(vector_store_path)
        self.model_provider = model_provider
        self.model_name = model_name
        
        # Initialize LLM client
        if model_provider == "anthropic" and anthropic:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
                logger.info(f"✅ Initialized Anthropic client for patient interactions")
            else:
                logger.warning("⚠️  ANTHROPIC_API_KEY not found. Patient RAG requires LLM.")
                self.client = None
        elif model_provider == "openai" and openai:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
                logger.info(f"✅ Initialized OpenAI client for patient interactions")
            else:
                logger.warning("⚠️  OPENAI_API_KEY not found. Patient RAG requires LLM.")
                self.client = None
        else:
            logger.warning(f"⚠️  LLM provider not available. Patient RAG requires LLM for empathetic responses.")
            self.client = None

    def _clean_text(self, text: str) -> str:
        """Clean extracted text from protocols."""
        if not text:
            return ""
        
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'[^\w\s.,!?;:()\-\'"/$%&]', ' ', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()

    def _detect_patient_concern_type(self, message: str) -> str:
        """Detect the type of patient concern to tailor the response approach."""
        message_lower = message.lower()
        
        # Emotional/anxiety concerns
        if any(word in message_lower for word in [
            "worried", "scared", "anxious", "frightened", "concerned", "nervous",
            "afraid", "stress", "overwhelming", "confused", "lost", "helpless"
        ]):
            return "emotional_support"
        
        # Symptom-related
        elif any(word in message_lower for word in [
            "symptoms", "pain", "hot flashes", "night sweats", "mood", "sleep",
            "bleeding", "weight", "energy", "fatigue", "brain fog"
        ]):
            return "symptom_guidance"
        
        # Treatment/medication questions
        elif any(word in message_lower for word in [
            "treatment", "medication", "hormone", "therapy", "side effects",
            "options", "alternatives", "help", "what can i do"
        ]):
            return "treatment_options"
        
        # Lifestyle/self-care
        elif any(word in message_lower for word in [
            "diet", "exercise", "lifestyle", "what should i", "how can i",
            "tips", "advice", "daily", "routine", "habits"
        ]):
            return "lifestyle_guidance"
        
        # Appointment/process questions
        elif any(word in message_lower for word in [
            "appointment", "visit", "test", "lab", "when", "how long",
            "process", "next steps", "follow up"
        ]):
            return "process_guidance"
        
        return "general_support"

    def _enhance_patient_query(self, message: str, concern_type: str) -> List[str]:
        """Create search queries tailored to patient concerns."""
        
        base_queries = [message]
        
        # Add concern-specific expansions
        if concern_type == "symptom_guidance":
            base_queries.extend([
                f"patient education {message}",
                f"symptom management {message}",
                f"what to expect {message}",
                f"normal symptoms perimenopause menopause"
            ])
        
        elif concern_type == "treatment_options":
            base_queries.extend([
                f"treatment options {message}",
                f"patient considerations {message}",
                f"shared decision making {message}",
                f"alternatives therapy options"
            ])
        
        elif concern_type == "lifestyle_guidance":
            base_queries.extend([
                f"lifestyle recommendations {message}",
                f"self-care {message}",
                f"patient tips {message}",
                f"daily habits wellness"
            ])
        
        elif concern_type == "emotional_support":
            base_queries.extend([
                f"patient support {message}",
                f"emotional wellness {message}",
                f"coping strategies {message}",
                f"mental health menopause"
            ])
        
        return base_queries

    def _search_for_patient_guidance(self, message: str, concern_type: str, k: int = 8) -> List[tuple]:
        """Search for relevant patient guidance from protocols."""
        
        search_queries = self._enhance_patient_query(message, concern_type)
        all_results = {}
        
        for i, query in enumerate(search_queries):
            try:
                results = self.vector_store.search_with_scores(
                    query=query,
                    k=k,
                    filter_dict=None  # Search all protocol types for patient guidance
                )
                
                weight = 1.0 if i == 0 else 0.8
                
                for doc, score in results:
                    doc_id = str(hash(doc.page_content[:100]))
                    if doc_id in all_results:
                        existing_score = all_results[doc_id][1]
                        all_results[doc_id] = (doc, max(existing_score, score * weight) + 0.1)
                    else:
                        all_results[doc_id] = (doc, score * weight)
                        
            except Exception as e:
                logger.warning(f"Patient search failed for query '{query}': {str(e)}")
                continue
        
        combined_results = list(all_results.values())
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results[:k]

    def chat(self, 
             message: str, 
             patient_context: Optional[Dict[str, Any]] = None,
             conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Main chat interface for patient interactions."""
        
        if not self.client:
            return {
                "response": "I'm sorry, but I'm currently unable to provide personalized guidance. Please reach out to your Midi care team directly for support.",
                "sources": [],
                "concern_type": "system_error"
            }
        
        try:
            # Detect the type of concern
            concern_type = self._detect_patient_concern_type(message)
            
            # Search for relevant protocol information
            relevant_docs = self._search_for_patient_guidance(message, concern_type)
            
            # Generate empathetic response
            response = self._generate_patient_response(
                message, 
                relevant_docs, 
                concern_type, 
                patient_context, 
                conversation_history
            )
            
            return {
                "response": response,
                "sources": [self._format_patient_source(doc) for doc, _ in relevant_docs[:3]],
                "concern_type": concern_type
            }
            
        except Exception as e:
            logger.error(f"Error in patient chat: {str(e)}")
            return {
                "response": "I apologize, but I'm having trouble right now. Your Midi care team is always here to help - please don't hesitate to reach out to them directly.",
                "sources": [],
                "concern_type": "system_error"
            }

    def _generate_patient_response(self, 
                                 message: str,
                                 relevant_docs: List[tuple],
                                 concern_type: str,
                                 patient_context: Optional[Dict[str, Any]] = None,
                                 conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate empathetic, patient-focused responses."""
        
        # Build context from protocols
        clinical_context = ""
        if relevant_docs:
            cleaned_context_parts = []
            for i, (doc, score) in enumerate(relevant_docs[:4]):
                cleaned_content = self._clean_text(doc.page_content)
                cleaned_context_parts.append(f"Protocol Information {i+1}:\n{cleaned_content}")
            clinical_context = "\n\n".join(cleaned_context_parts)
        
        # Build patient context
        patient_info = ""
        if patient_context:
            patient_info = f"Patient Context: {', '.join([f'{k}: {v}' for k, v in patient_context.items()])}\n"
        
        # Build conversation context
        conversation_context = ""
        if conversation_history:
            recent_messages = conversation_history[-3:]  # Last 3 exchanges
            conversation_context = "Recent Conversation:\n" + "\n".join([
                f"Patient: {msg.get('patient', '')}\nMidi: {msg.get('assistant', '')}" 
                for msg in recent_messages
            ]) + "\n"
        
        # Determine if this is early in conversation (for initial affirmation)
        is_early_conversation = not conversation_history or len(conversation_history) <= 1
        
        # Create empathy guidance based on concern type
        empathy_guidance = self._get_empathy_guidance(concern_type, is_early_conversation)
        
        prompt = f"""You are MidiChat, Midi Health's AI assistant. You provide helpful, evidence-based information to support women through their health journey.

{patient_info}{conversation_context}

Clinical Protocol Information:
{clinical_context}

Patient Message: "{message}"

Concern Type: {concern_type}
Early Conversation: {is_early_conversation}

CORE PRINCIPLES FOR YOUR RESPONSE:

{empathy_guidance}

RESPONSE GUIDELINES:
1. BE HELPFUL: Focus on providing useful, actionable information
2. BE CONVERSATIONAL: Use warm but not overly effusive language
3. PROVIDE EVIDENCE-BASED GUIDANCE: Use protocol information to offer practical help
4. BE CLEAR: Use accessible language (not clinical jargon)
5. EMPOWER: Help them feel informed and capable
6. FOCUS ON EDUCATION: Give them useful information they can act on
7. SUGGEST NEXT STEPS: Offer concrete actions or additional questions to explore
8. MENTION CARE TEAM SPARINGLY: Only suggest scheduling when they specifically ask about appointments or have complex medical situations

TONE & STYLE:
- Helpful and informative
- Warm but professional (not overly effusive)
- If early conversation: Include one brief welcoming acknowledgment
- If ongoing conversation: Skip excessive affirmations, focus on being helpful
- Be encouraging but not repetitive with praise

STRUCTURE YOUR RESPONSE:
- Brief acknowledgment (1 sentence, only if early conversation)
- Helpful information/guidance (main content with specific, actionable advice)
- Natural wrap-up
- DO NOT include "You might also want to ask" suggestions - these are annoying

AVOID:
- Excessive affirmations like "Thank you for...", "I understand...", "It's wonderful that..."
- Repetitive praise or validation in every response
- Ending with "You might also want to ask" or similar suggestion lists
- Being overly effusive or obsequious

Provide a helpful, informative response:"""

        try:
            if self.model_provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=800,  # Conversational length
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            
            elif self.model_provider == "openai":
                openai_model = "gpt-4" if "gpt-4" in self.model_name else "gpt-3.5-turbo"
                response = self.client.chat.completions.create(
                    model=openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=800
                )
                return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating patient response: {str(e)}")
            return self._fallback_patient_response(concern_type)

    def _get_empathy_guidance(self, concern_type: str, is_early_conversation: bool = False) -> str:
        """Get specific empathy guidance based on concern type."""
        
        if is_early_conversation:
            early_guidance = """
- BRIEF WELCOME: Include one welcoming acknowledgment (1 sentence)
- FOCUS ON BEING HELPFUL: Get straight to providing useful information"""
        else:
            early_guidance = """
- SKIP EXCESSIVE AFFIRMATIONS: Don't repeat "Thank you for asking" or similar
- FOCUS ON BEING HELPFUL: Get straight to providing useful information"""
        
        guidance_map = {
            "emotional_support": f"""{early_guidance}
- Address their emotional concerns directly
- Provide practical coping strategies
- Normalize their experience briefly without overdoing it""",
            
            "symptom_guidance": f"""{early_guidance}
- Acknowledge symptoms are real and manageable
- Provide practical management strategies
- Give hope with concrete solutions""",
            
            "treatment_options": f"""{early_guidance}
- Address their desire to explore options
- Provide clear information about available treatments
- Emphasize evidence-based choices""",
            
            "lifestyle_guidance": f"""{early_guidance}
- Support their proactive approach
- Provide specific, actionable advice
- Focus on manageable steps""",
            
            "process_guidance": f"""{early_guidance}
- Clarify confusing processes
- Provide step-by-step guidance
- Make complex things simple""",
            
            "general_support": f"""{early_guidance}
- Be helpful and informative
- Provide relevant information
- Keep response focused and useful"""
        }
        
        return guidance_map.get(concern_type, guidance_map["general_support"])

    def _fallback_patient_response(self, concern_type: str) -> str:
        """Provide empathetic fallback when LLM fails."""
        
        responses = {
            "emotional_support": "What you're feeling is completely valid. Many women experience similar concerns during this stage of life, and there are strategies that can help you feel more supported and resilient during this transition.",
            
            "symptom_guidance": "What you're experiencing is real and manageable. There are many evidence-based approaches - from lifestyle modifications to various treatment options - that can help you feel better.",
            
            "treatment_options": "There are many effective treatments available, ranging from lifestyle approaches to various medical therapies, each with their own benefits to consider.",
            
            "lifestyle_guidance": "Small, consistent changes in areas like nutrition, movement, sleep, and stress management can make a meaningful difference in how you feel.",
            
            "process_guidance": "Breaking things down into manageable steps and knowing what questions to ask can help you feel more confident moving forward.",
            
            "general_support": "There's helpful information available to support you on your health journey."
        }
        
        base_response = responses.get(concern_type, responses["general_support"])
        return f"{base_response}\n\nWhat specific aspects would you like to explore further?"

    def _generate_follow_up_suggestions(self, concern_type: str, message: str) -> List[str]:
        """Generate helpful follow-up suggestions for patients."""
        
        base_suggestions = {
            "emotional_support": [
                "Would you like to talk about what's been most challenging?",
                "Have you been able to connect with other women going through similar experiences?",
                "Would information about stress management techniques be helpful?",
                "What coping strategies have you tried so far?"
            ],
            
            "symptom_guidance": [
                "Would you like to track your symptoms to better understand patterns?",
                "Are there specific times of day when symptoms are more challenging?",
                "Would you like tips for managing symptoms day-to-day?",
                "What lifestyle factors seem to affect your symptoms?"
            ],
            
            "treatment_options": [
                "Would you like to learn more about different types of treatments available?",
                "Do you have questions about specific treatments you've heard about?",
                "Would information about what to expect from different options be helpful?",
                "What factors are most important to you when considering treatment?"
            ],
            
            "lifestyle_guidance": [
                "Would you like specific tips you can start with today?",
                "Are there particular areas of lifestyle you'd like to focus on?",
                "Would meal planning or exercise suggestions be helpful?",
                "What lifestyle changes feel most manageable to start with?"
            ],
            
            "process_guidance": [
                "Would you like help understanding what to expect next?",
                "Do you need clarification about any health topics?",
                "Would information about preparing questions be helpful?",
                "What aspects feel most confusing right now?"
            ]
        }
        
        return base_suggestions.get(concern_type, [
            "What other questions can I help you with?",
            "Is there anything specific you'd like to know more about?",
            "How else can I support you today?"
        ])
    
    def _format_patient_source(self, doc) -> str:
        """Format source information for patient-friendly display."""
        data_source = doc.metadata.get("data_source", "protocols")
        source_name = doc.metadata.get("source", "Midi Resource")
        
        # Patient-friendly source labels
        source_labels = {
            "protocols": "📋 Clinical Guidelines",
            "midi_blog_posts": "📰 Midi Health Article", 
            "midi_zendesk_articles": "❓ Midi Support Resource"
        }
        
        source_type = source_labels.get(data_source, "📄 Midi Resource")
        
        # Clean up filename for patient display
        clean_name = source_name.replace('_', ' ').replace('.pdf', '').replace('.md', '')
        if len(clean_name) > 50:
            clean_name = clean_name[:47] + "..."
            
        return f"{source_type}: {clean_name}"

    def get_stats(self) -> Dict[str, Any]:
        """Get patient RAG system statistics."""
        vector_stats = self.vector_store.get_collection_stats()
        return {
            "vector_store": vector_stats,
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "patient_focused": True,
            "llm_available": self.client is not None
        }