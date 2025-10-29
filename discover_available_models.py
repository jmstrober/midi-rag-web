#!/usr/bin/env python3
"""
Comprehensive Model Discovery Tool for Anthropic API
Discovers all available models with your current API key
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not available")

def test_model_list(client):
    """Try to get official model list if API supports it"""
    print("🔍 Attempting to get official model list...")
    
    try:
        # Some APIs have a models endpoint
        models = client.models.list()
        print("✅ Retrieved official model list:")
        for model in models.data:
            print(f"   - {model.id}")
        return [model.id for model in models.data]
    except Exception as e:
        print(f"ℹ️  Official model list not available: {e}")
        return None

def get_comprehensive_model_list():
    """Generate comprehensive list of potential Claude model names to test"""
    
    # Known Claude model families and versions
    models = [
        # Claude 4 series (newest)
        "claude-4",
        "claude-4-opus",
        "claude-4-sonnet", 
        "claude-4-haiku",
        "claude-sonnet-4",
        "claude-opus-4",
        "claude-haiku-4",
        
        # Claude 3.5 series
        "claude-3-5-opus",
        "claude-3-5-sonnet",
        "claude-3-5-haiku",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-5-sonnet-latest",
        "claude-3-5-opus-20241022",
        "claude-3-5-haiku-20241022",
        
        # Claude 3 series (known working models)
        "claude-3-opus",
        "claude-3-sonnet", 
        "claude-3-haiku",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-3-haiku-20240307-v1:0",
        
        # Alternative naming patterns
        "claude-sonnet-3-5",
        "claude-sonnet-3-7",
        "claude-opus-3-5",
        "claude-haiku-3-5",
        
        # Generic aliases
        "claude-latest",
        "claude-best",
        "claude",
        
        # Bedrock-style names (if using AWS)
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic.claude-3-opus-20240229-v1:0",
        "anthropic.claude-3-5-sonnet-20240620-v1:0",
        
        # Potential new models
        "claude-instant",
        "claude-instant-v1",
        "claude-instant-1.2",
        "claude-2.1",
        "claude-2.0",
        "claude-1.3",
    ]
    
    return models

def test_single_model(client, model_name, quick_test=True):
    """Test a single model with minimal resource usage"""
    
    try:
        # Use minimal token count for quick testing
        max_tokens = 5 if quick_test else 50
        test_message = "Hi" if quick_test else "Please respond with 'Model accessible' to confirm you're working."
        
        response = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            messages=[{
                "role": "user", 
                "content": test_message
            }]
        )
        
        response_text = response.content[0].text.strip()
        return {
            "status": "success",
            "response": response_text,
            "error": None
        }
        
    except Exception as e:
        error_msg = str(e)
        error_type = "unknown"
        
        # Categorize errors
        if "404" in error_msg or "not_found" in error_msg:
            error_type = "not_found"
        elif "401" in error_msg or "unauthorized" in error_msg:
            error_type = "unauthorized"
        elif "403" in error_msg or "forbidden" in error_msg:
            error_type = "forbidden"
        elif "429" in error_msg or "rate_limit" in error_msg:
            error_type = "rate_limited"
        elif "400" in error_msg or "bad_request" in error_msg:
            error_type = "bad_request"
        
        return {
            "status": "failed",
            "response": None,
            "error": error_msg,
            "error_type": error_type
        }

def discover_available_models():
    """Main discovery function"""
    print("🔬 COMPREHENSIVE CLAUDE MODEL DISCOVERY")
    print("=" * 80)
    
    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment variables")
        return None
    
    print(f"✅ API Key found: {api_key[:20]}...{api_key[-4:]}")
    
    # Initialize client
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        print("✅ Anthropic client initialized")
    except ImportError:
        print("❌ Anthropic package not installed. Run: pip install anthropic")
        return None
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return None
    
    # Try to get official model list first
    official_models = test_model_list(client)
    
    # Get comprehensive test list
    test_models = get_comprehensive_model_list()
    
    print(f"\n🧪 Testing {len(test_models)} potential model names...")
    print("(This may take a few minutes - testing with minimal tokens)")
    print("-" * 80)
    
    # Track results
    results = {
        "working_models": [],
        "failed_models": {},
        "test_timestamp": datetime.now().isoformat(),
        "api_key_prefix": f"{api_key[:20]}...{api_key[-4:]}"
    }
    
    # Test each model
    for i, model in enumerate(test_models, 1):
        print(f"[{i:2d}/{len(test_models)}] Testing: {model:<40}", end=" ")
        
        result = test_single_model(client, model, quick_test=True)
        
        if result["status"] == "success":
            print("✅ SUCCESS")
            results["working_models"].append({
                "name": model,
                "response": result["response"]
            })
        else:
            print(f"❌ {result['error_type'].upper()}")
            if result["error_type"] not in results["failed_models"]:
                results["failed_models"][result["error_type"]] = []
            results["failed_models"][result["error_type"]].append(model)
        
        # Brief pause to avoid rate limiting
        import time
        time.sleep(0.1)
    
    return results

def generate_detailed_report(results):
    """Generate a detailed report of findings"""
    
    print("\n" + "=" * 80)
    print("📊 DETAILED DISCOVERY REPORT")
    print("=" * 80)
    
    # Working models
    if results["working_models"]:
        print(f"\n✅ WORKING MODELS ({len(results['working_models'])} found):")
        print("-" * 50)
        for model in results["working_models"]:
            print(f"🎯 {model['name']}")
            print(f"   Response: '{model['response']}'")
            print()
        
        # Recommend best model
        print("🏆 RECOMMENDATIONS:")
        for model in results["working_models"]:
            name = model['name']
            if 'sonnet' in name.lower():
                print(f"   🥇 BEST: {name} (Sonnet - balanced performance)")
                break
            elif 'opus' in name.lower():
                print(f"   🥇 BEST: {name} (Opus - highest capability)")
                break
            elif 'haiku' in name.lower():
                print(f"   🥈 GOOD: {name} (Haiku - fast and efficient)")
                
    else:
        print("\n❌ NO WORKING MODELS FOUND")
        print("This suggests an issue with your API key or account access.")
    
    # Failed models breakdown
    print(f"\n📋 FAILED MODELS BREAKDOWN:")
    print("-" * 50)
    for error_type, models in results["failed_models"].items():
        print(f"\n{error_type.upper()} ({len(models)} models):")
        if error_type == "not_found":
            print("   → These models don't exist or aren't available in your tier")
        elif error_type == "unauthorized":
            print("   → API key authentication issues")
        elif error_type == "forbidden":
            print("   → Access denied (may require higher tier)")
        elif error_type == "rate_limited":
            print("   → Rate limit reached during testing")
        
        # Show first few examples
        for model in models[:5]:
            print(f"     - {model}")
        if len(models) > 5:
            print(f"     ... and {len(models)-5} more")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_discovery_report_{timestamp}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Full report saved to: {filename}")
    except Exception as e:
        print(f"\n⚠️  Could not save report: {e}")
    
    return results

def test_working_models_detailed(results):
    """Test working models with a more detailed medical query"""
    
    if not results["working_models"]:
        return
    
    print("\n" + "=" * 80)
    print("🏥 DETAILED MEDICAL QUERY TESTING")
    print("=" * 80)
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    
    medical_query = """Based on recent medical research, can you provide guidance for a patient who asks: 
    'I have a family history of breast cancer. Can I still consider hormone replacement therapy?'
    Please provide a balanced, evidence-based response."""
    
    for model_info in results["working_models"]:
        model_name = model_info["name"]
        print(f"\n🧪 Testing {model_name} with medical query...")
        
        result = test_single_model(client, model_name, quick_test=False)
        
        if result["status"] == "success":
            response = result["response"]
            print(f"✅ Response length: {len(response)} characters")
            print(f"📝 Sample: {response[:200]}...")
            
            # Check quality indicators
            quality_score = 0
            if len(response) > 100:
                quality_score += 1
            if any(term in response.lower() for term in ['hormone', 'breast cancer', 'family history']):
                quality_score += 1
            if any(term in response.lower() for term in ['risk', 'doctor', 'medical']):
                quality_score += 1
            
            print(f"📊 Quality score: {quality_score}/3")
        else:
            print(f"❌ Failed: {result['error']}")

if __name__ == "__main__":
    print("🚀 Starting comprehensive Claude model discovery...")
    print("This will test many potential model names to find what works with your API key.\n")
    
    # Run discovery
    results = discover_available_models()
    
    if results:
        # Generate report
        generate_detailed_report(results)
        
        # Test working models with medical queries
        test_working_models_detailed(results)
        
        print("\n" + "=" * 80)
        print("🎯 NEXT STEPS:")
        if results["working_models"]:
            best_model = results["working_models"][0]["name"]
            print(f"1. Use this model in your RAG engine: '{best_model}'")
            print("2. Update both patient_rag_engine.py and rag_engine.py")
            print("3. Restart your Streamlit app")
            print("4. Test with your breast cancer/HRT question")
        else:
            print("1. Check your Anthropic account status")
            print("2. Verify your API key has proper permissions")
            print("3. Consider contacting Anthropic support")
            print("4. Try using OpenAI as an alternative")
        
        print("\n📧 If you need help, share the generated JSON report with technical support.")
    
    print("\n" + "=" * 80)