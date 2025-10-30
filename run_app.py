#!/usr/bin/env python3
"""
Safe launcher for the Streamlit app with conservative settings
"""

import os
import sys
import multiprocessing

# Set conservative multiprocessing settings before any imports
if __name__ == "__main__":
    # Force spawn method for multiprocessing to avoid fork issues
    multiprocessing.set_start_method('spawn', force=True)
    
    # Set all the environment variables early
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    # PyTorch specific settings
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    
    # Disable CUDA if available
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    print("🚀 Starting MIDI RAG with conservative multiprocessing settings...")
    
    # Now import and configure torch
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if hasattr(torch, 'set_default_device'):
            torch.set_default_device('cpu')
        if hasattr(torch, 'set_default_dtype'):
            torch.set_default_dtype(torch.float32)
        # Disable MPS (Metal Performance Shaders) on macOS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.backends.mps.is_available = lambda: False
        print("✅ PyTorch configured for CPU-only operation")
    except ImportError:
        print("⚠️ PyTorch not available")
    
    # Now run streamlit using the current Python executable (from venv)
    import subprocess
    
    # Get the current Python executable (should be from venv)
    python_exe = sys.executable
    print(f"Using Python executable: {python_exe}")
    
    # Verify we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
    else:
        print("⚠️ WARNING: Not running in virtual environment!")
    
    result = subprocess.run([
        python_exe, '-m', 'streamlit', 'run', 'streamlit_app.py'
    ], cwd=os.getcwd(), env=os.environ.copy())
    
    sys.exit(result.returncode)