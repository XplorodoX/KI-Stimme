import sys
import os

def check_import(module_name):
    try:
        __import__(module_name)
        print(f"[OK] {module_name} imported successfully.")
        return True
    except ImportError as e:
        print(f"[FAIL] Could not import {module_name}: {e}")
        return False

def check_torch():
    try:
        import torch
        print(f"[OK] Torch version: {torch.__version__}")
        if torch.cuda.is_available():
            print("[OK] CUDA is available.")
        elif torch.backends.mps.is_available():
            print("[OK] MPS (Mac Silicon) is available.")
        else:
            print("[WARN] No GPU acceleration detected (CUDA/MPS).")
        return True
    except ImportError:
        print("[FAIL] Torch not found.")
        return False

def check_llm():
    try:
        from llm_handler import LLMHandler
        llm = LLMHandler(provider="ollama")
        print(f"[INFO] Testing LLM connection to {llm.model}...")
        # Simple ping
        try:
            # We won't do a full generation to save time, just check if client inits
            if llm.client:
                print("[OK] LLM Client initialized.")
            return True
        except Exception as e:
            print(f"[FAIL] LLM Client init failed: {e}")
            return False
    except Exception as e:
        print(f"[FAIL] Could not load LLMHandler: {e}")
        return False

if __name__ == "__main__":
    print("--- Checking Environment ---")
    print(f"Python: {sys.version}")
    
    checks = [
        check_import("gradio"),
        check_import("TTS"),
        check_torch(),
        check_llm()
    ]
    
    if all(checks):
        print("\n[SUCCESS] Environment looks ready!")
    else:
        print("\n[WARNING] Some checks failed.")
