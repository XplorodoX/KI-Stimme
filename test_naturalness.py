import os
import logging
from llm_handler import LLMHandler
from voice_cloner import VoiceCloner
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_natural_generation():
    print("=== Testing Natural Text Generation ===")
    
    # Initialize LLM
    try:
        llm = LLMHandler()
        
        # Test prompt
        topic = "Warum ist der Himmel blau?"
        prompt = f"Erkläre mir kurz: {topic}"
        
        print(f"\nUser Request: {prompt}")
        print("-" * 30)
        
        # Generate text using the new system prompt
        # We need to manually pass the system prompt if we want to verify it, 
        # but LLMHandler uses config.SYSTEM_PROMPT by default which we updated.
        
        # Let's explicitly get the German prompt to be sure
        sys_prompt = config.get_system_prompt("de")
        print(f"System Prompt used:\n{sys_prompt}\n")
        
        text = llm.generate_text(prompt, system_prompt=sys_prompt)
        
        print("-" * 30)
        print(f"Generated Script:\n{text}")
        print("-" * 30)
        
        return text
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return None

def test_tts_generation(text, ref_audio_path):
    if not text:
        print("No text to synthesize.")
        return

    if not os.path.exists(ref_audio_path):
        print(f"Reference audio not found at {ref_audio_path}. Skipping TTS test.")
        return

    print("\n=== Testing Natural TTS ===")
    try:
        cloner = VoiceCloner()
        output_path = "test_natural_output.wav"
        
        # Use the new default parameters from config
        cloner.clone_voice(
            text=text,
            speaker_wav_path=ref_audio_path,
            language="de",
            file_path=output_path
        )
        print(f"Generated audio saved to: {output_path}")
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")

if __name__ == "__main__":
    # 1. Generate Text
    generated_text = test_natural_generation()
    
    # 2. Synthesize (optional, requires a reference file)
    # Replace this with a real path if you have one
    reference_audio = "sample_speaker.wav" 
    
    # Check if we have any wav in outputs to use as fallback for testing (though not ideal)
    if not os.path.exists(reference_audio):
        # Try to find a recent output file just to test the pipeline
        import glob
        outputs = sorted(glob.glob("outputs/*.wav"))
        if outputs:
            reference_audio = outputs[-1]
            print(f"Using most recent output as reference for testing: {reference_audio}")
    
    test_tts_generation(generated_text, reference_audio)
