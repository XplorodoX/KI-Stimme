"""TTS Server using Microsoft Edge TTS - Much more natural sounding voices!"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import logging
import asyncio
import edge_tts
import os
from pathlib import Path
from datetime import datetime
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Edge TTS Server", description="Natural sounding TTS using Microsoft Edge voices")

# Output directory
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Available German voices (male & female options)
VOICES = {
    # Male voices - natural and friendly
    "male": "de-DE-FlorianMultilingualNeural",  # Cheerful, Warm - BEST male voice!
    "male_alt": "de-DE-ConradNeural",            # Engaging, Friendly
    "male_casual": "de-AT-JonasNeural",          # Light-Hearted, Whimsical
    
    # Female voices  
    "female": "de-DE-SeraphinaMultilingualNeural",  # Casual
    "female_alt": "de-DE-AmalaNeural",              # Well-Rounded, Animated
    "female_calm": "de-DE-KatjaNeural",             # Calm, Pleasant
}

# Default voice - Florian sounds very natural and friendly
DEFAULT_VOICE = VOICES["male"]

# Emotion to voice rate/pitch mapping for more expressive speech
EMOTION_SETTINGS = {
    "neutral": {"rate": "+0%", "pitch": "+0Hz"},
    "happy": {"rate": "+10%", "pitch": "+50Hz"},
    "sad": {"rate": "-15%", "pitch": "-30Hz"},
    "angry": {"rate": "+20%", "pitch": "+20Hz"},
    "calm": {"rate": "-10%", "pitch": "-10Hz"},
    "excited": {"rate": "+25%", "pitch": "+40Hz"},
    "thinking": {"rate": "-5%", "pitch": "+0Hz"},
    "surprised": {"rate": "+15%", "pitch": "+60Hz"},
}

class TTSRequest(BaseModel):
    text: str
    language: str = "de"
    emotion: str = "neutral"
    voice: str = None  # Optional: override voice
    rate: str = None   # Optional: override rate like "+10%" or "-5%"
    pitch: str = None  # Optional: override pitch like "+50Hz"

@app.post("/tts")
async def generate_speech(request: TTSRequest):
    """Generate natural sounding speech using Edge TTS."""
    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Select voice
        voice = request.voice or DEFAULT_VOICE
        
        # Get emotion settings
        emotion = request.emotion.lower() if request.emotion else "neutral"
        settings = EMOTION_SETTINGS.get(emotion, EMOTION_SETTINGS["neutral"])
        
        # Allow overrides
        rate = request.rate or settings["rate"]
        pitch = request.pitch or settings["pitch"]
        
        logger.info(f"Generating TTS: voice={voice}, emotion={emotion}, rate={rate}, pitch={pitch}")
        logger.info(f"Text: {text[:100]}...")
        
        # Generate audio with edge-tts
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate,
            pitch=pitch
        )
        
        # Collect audio bytes
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        
        if not audio_data:
            raise HTTPException(status_code=500, detail="No audio generated")
        
        # Save to file for debugging (optional)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"edge_tts_{timestamp}.mp3"
        with open(output_path, "wb") as f:
            f.write(audio_data)
        
        logger.info(f"Generated {len(audio_data)} bytes of audio -> {output_path}")
        
        # Return audio directly as MP3
        return Response(
            content=audio_data,
            media_type="audio/mpeg",
            headers={"Content-Disposition": f"inline; filename=speech.mp3"}
        )
        
    except Exception as e:
        logger.error(f"TTS Generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/voices")
async def list_voices():
    """List available voice options."""
    return {
        "default": DEFAULT_VOICE,
        "available": VOICES,
        "emotions": list(EMOTION_SETTINGS.keys())
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "engine": "edge-tts", "default_voice": DEFAULT_VOICE}

if __name__ == "__main__":
    print("🎙️ Starting Edge TTS Server...")
    print(f"📢 Default voice: {DEFAULT_VOICE}")
    print(f"🎭 Emotions: {list(EMOTION_SETTINGS.keys())}")
    uvicorn.run(app, host="0.0.0.0", port=8001)
