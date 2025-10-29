#!/usr/bin/env python3
"""
Test Claude Sonnet 4 API connection and model availability
"""

import os
import sys
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not available")

def test_anthropic_connection():
    """Test the Anthropic API connection and model availability."""
    print("🧪 Testing Anthropic Claude API Connection...")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment variables")
        print("Please set your Anthropic API key in the .env file")
        return False
    
    print(f"✅ API Key found: {api_key[:20]}...{api_key[-4:]}")
    
    # Test anthropic package
    try:
        import anthropic
        print("✅ Anthropic package imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import anthropic package: {e}")
        print("Install with: pip install anthropic")
        return False
    
    # Initialize client
    try:
        client = anthropic.Anthropic(api_key=api_key)
        print("✅ Anthropic client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Anthropic client: {e}")
        return False
    
    # Test different model names to find what works
    models_to_test = [
        "claude-sonnet-4",                    # What we're trying to use
        "claude-sonnet-3-7",                 # Testing this specific name
        "claude-3-5-sonnet-20241022",        # Latest version
        "claude-3-5-sonnet-20240620",        # Previous version
        "claude-3-5-sonnet-latest",          # Latest alias
        "claude-3-sonnet-20240229",          # Older stable version
        "claude-3-haiku-20240307",           # Smaller model
    ]
    
    working_models = []
    
    for model in models_to_test:
        print(f"\n🔍 Testing model: {model}")
        try:
            response = client.messages.create(
                model=model,
                max_tokens=50,
                messages=[{
                    "role": "user", 
                    "content": "Hello! Please respond with just 'Model working' to confirm you're accessible."
                }]
            )
            
            response_text = response.content[0].text.strip()
            print(f"✅ {model} - SUCCESS")
            print(f"   Response: {response_text}")
            working_models.append(model)
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ {model} - FAILED: {error_msg}")
            
            # Check for specific error types
            if "404" in error_msg or "not_found" in error_msg:
                print("   → Model not available in your API access")
            elif "401" in error_msg or "unauthorized" in error_msg:
                print("   → API key authentication issue")
            elif "429" in error_msg or "rate_limit" in error_msg:
                print("   → Rate limit reached")
            else:
                print(f"   → Other error: {error_msg}")
    
    print("\n" + "=" * 50)
    print("📊 SUMMARY:")
    if working_models:
        print(f"✅ Found {len(working_models)} working models:")
        for model in working_models:
            print(f"   - {model}")
        print(f"\n🎯 RECOMMENDED: Use '{working_models[0]}' as your model name")
        return working_models[0]
    else:
        print("❌ No working models found!")
        print("Check your API key and account permissions")
        return None

def test_medical_query_with_model(model_name):
    """Test a medical query with the working model."""
    print(f"\n🏥 Testing medical query with {model_name}...")
    print("=" * 50)
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    
    # Test with sample medical documents (simulating RAG context)
    medical_context = """
    Based on recent medical research:
    
    1. From "Midi's Definitive Guide to Menopause Treatment for Breast Cancer Survivors":
    The relationship between HRT and breast cancer has been misunderstood. Recent studies show that 
    the 2002 WHI findings were exaggerated. For women with family history but no personal history 
    of breast cancer, HRT can be considered with proper monitoring.
    
    2. From Clinical Protocol for Hormone Therapy:
    Women with family history of breast cancer should be evaluated individually. Risk factors include 
    BRCA mutations, age at first pregnancy, and density of breast tissue.
    """
    
    query = "I have a family history of breast cancer. Can I still do HRT?"
    
    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": f"""You are a medical AI assistant. Based on this medical information:

{medical_context}

Please answer this patient question: {query}

Provide a helpful, informative response based on the medical information provided."""
            }]
        )
        
        response_text = response.content[0].text.strip()
        print(f"✅ Medical query successful!")
        print(f"Query: {query}")
        print(f"Response: {response_text}")
        
        # Check if response is substantive (not generic)
        if len(response_text) > 100 and any(term in response_text.lower() for term in ['hrt', 'hormone', 'family history', 'breast cancer']):
            print("✅ Response appears to be medically relevant and substantive")
            return True
        else:
            print("⚠️ Response seems generic or too short")
            return False
            
    except Exception as e:
        print(f"❌ Medical query failed: {e}")
        return False

if __name__ == "__main__":
    print("🔬 CLAUDE SONNET 4 CONNECTION TEST")
    print("=" * 60)
    
    # Test basic connection
    working_model = test_anthropic_connection()
    
    if working_model:
        # Test medical query
        test_medical_query_with_model(working_model)
        
        print("\n🎯 NEXT STEPS:")
        print(f"1. Update your RAG engines to use model: '{working_model}'")
        print("2. Restart your Streamlit app")
        print("3. Test your breast cancer/HRT question again")
    else:
        print("\n💡 TROUBLESHOOTING:")
        print("1. Verify your ANTHROPIC_API_KEY is correct")
        print("2. Check your Anthropic account has API access")
        print("3. Ensure you have credits/usage allowance")
        print("4. Try using OpenAI as an alternative")
    
    print("\n" + "=" * 60)