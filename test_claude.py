#!/usr/bin/env python3
"""
Test Claude API connection and model availability
"""

import os
from dotenv import load_dotenv
load_dotenv()

try:
    import anthropic
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment")
        exit(1)
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Test different model names
    models_to_test = [
        "claude-3-5-sonnet-20241022",  # The one causing errors
        "claude-3-5-sonnet-20240620",  # Previous version
        "claude-3-5-sonnet-latest",    # Latest alias
        "claude-3-sonnet-20240229",    # Older but stable
    ]
    
    for model in models_to_test:
        try:
            print(f"🧪 Testing model: {model}")
            response = client.messages.create(
                model=model,
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello, just testing if this model works. Please respond briefly."}]
            )
            print(f"✅ {model} - SUCCESS")
            print(f"   Response: {response.content[0].text[:50]}...")
            break  # Use the first working model
        except Exception as e:
            print(f"❌ {model} - FAILED: {str(e)}")
    
except ImportError:
    print("❌ anthropic package not installed")
except Exception as e:
    print(f"❌ Error: {e}")