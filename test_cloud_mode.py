#!/usr/bin/env python3
"""
Test script to simulate Streamlit Cloud environment
"""
import warnings
warnings.filterwarnings('ignore')
import sys
import os

# Simulate Streamlit Cloud environment
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['HOSTNAME'] = 'streamlit-app-xyz'

# Add src to path
sys.path.append('src')

def test_cloud_mode():
    print("🔍 Testing Streamlit Cloud environment simulation...")
    
    try:
        from utils.VectorStoreManager import VectorStoreManager
        
        # Initialize VectorStoreManager (which should detect cloud mode)
        manager = VectorStoreManager(persist_directory="data/chroma_db")
        print("✅ VectorStoreManager initialized successfully in cloud mode")
        
        # Check if we're in online mode (should not have offline env vars set)
        offline_vars = ['TRANSFORMERS_OFFLINE', 'HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE']
        for var in offline_vars:
            if os.environ.get(var) == '1':
                print(f"❌ {var} is set to '1' - should be unset in cloud mode")
                return False
            else:
                print(f"✅ {var} is not set (correct for cloud mode)")
        
        print("✅ Cloud mode configuration is correct")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing Streamlit Cloud mode simulation...")
    
    success = test_cloud_mode()
    
    if success:
        print("\n🎉 Streamlit Cloud simulation test passed!")
    else:
        print("\n❌ Streamlit Cloud simulation test failed.")