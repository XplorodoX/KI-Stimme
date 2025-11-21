import os
import torch
from TTS.api import TTS

# Monkey patch torch.load to default weights_only=False for Coqui TTS compatibility
# (Even with torch 2.4.0, it's good practice if they backported it, but 2.4.0 shouldn't have it as default True yet)
# We'll keep it to be safe or remove it if it causes issues. 
# Actually, torch 2.4.0 DOES NOT have weights_only=True by default, so we can try without patching first
# to see if that was causing issues, OR keep it. 
# Let's keep the patch but make it robust.

try:
    _original_load = torch.load
    def safe_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
             # Only add if the function supports it
             # But safe to add False if we are sure.
             # Actually, let's just try loading NORMALLY first.
             pass
        return _original_load(*args, **kwargs)
    # torch.load = safe_load
except Exception as e:
    print(f"Patch failed: {e}")

print("Loading TTS...")
try:
    os.environ["COQUI_TOS_AGREED"] = "1"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("mps")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    import traceback
    traceback.print_exc()
