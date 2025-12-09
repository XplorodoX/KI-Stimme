from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse, JSONResponse
import logging
import config
from voice_cloner import VoiceCloner
import base64
import os
import uvicorn
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize VoiceCloner once on startup
try:
    logger.info("Initializing VoiceCloner...")
    cloner = VoiceCloner()
    logger.info("VoiceCloner ready!")
except Exception as e:
    logger.error(f"Failed to initialize VoiceCloner: {e}")
    cloner = None

class TTSRequest(BaseModel):
    text: str
    language: str = "de"
    emotion: str = "neutral"
    # Optional parameters
    speaker_wav: str = None 

# Default reference voice path - adjust this to your preferred default file
DEFAULT_SPEAKER_WAV = "test_natural_output.wav" 

@app.post("/tts")
async def generate_speech(request: TTSRequest):
    if not cloner:
        raise HTTPException(status_code=500, detail="VoiceCloner not initialized")

    try:
        # Determine speaker wav
        speaker_wav = request.speaker_wav or DEFAULT_SPEAKER_WAV
        if not os.path.exists(speaker_wav):
            # Fallback check output dir if not found in root
            potential_path = os.path.join("outputs", speaker_wav)
            if os.path.exists(potential_path):
                speaker_wav = potential_path
            else:
                 # Last resort: try to find ANY wav file in outputs to use as reference if default is missing
                 wavs = list(Path("outputs").glob("*.wav"))
                 if wavs:
                     speaker_wav = str(wavs[0])
                     logger.warning(f"Default speaker wav not found, using valid fallback: {speaker_wav}")
                 else:
                    raise HTTPException(status_code=404, detail=f"Reference audio file not found: {speaker_wav}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"tts_api_{timestamp}.wav"
        output_path = config.OUTPUT_DIR / output_filename
        
        # Ensure output directory
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating TTS for text: {request.text[:50]}...")
        
        generated_file = cloner.clone_voice(
            text=request.text,
            speaker_wav_path=str(speaker_wav),
            language=request.language,
            emotion=request.emotion,
            file_path=str(output_path)
        )

        # Read file and return as base64 JSON (compatible with existing frontend logic)
        with open(generated_file, "rb") as audio_file:
            audio_content = audio_file.read()
            audio_base64 = base64.b64encode(audio_content).decode("utf-8")

        return JSONResponse(content={
            "audio": audio_base64,
            "text": request.text,
            "emotion": request.emotion,
            "file": str(generated_file)
        })

    except Exception as e:
        logger.error(f"TTS Generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "cloner_ready": cloner is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
