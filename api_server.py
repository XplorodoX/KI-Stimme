from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse, StreamingResponse
import logging
import config
from voice_cloner import VoiceCloner
from llm_handler import LLMHandler
import os
import uvicorn
from pathlib import Path
from typing import List, Optional
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="MetaHuman Backend API", description="API for LLM Chat and TTS Generation")

# --- Global Instances ---
cloner: Optional[VoiceCloner] = None
llm: Optional[LLMHandler] = None

# --- Initialization ---
@app.on_event("startup")
async def startup_event():
    global cloner, llm
    logger.info("Starting up backend services...")
    
    # Initialize VoiceCloner
    try:
        logger.info("Initializing VoiceCloner (Coqui XTTS)...")
        cloner = VoiceCloner()
        logger.info("VoiceCloner ready!")
    except Exception as e:
        logger.error(f"Failed to initialize VoiceCloner: {e}")
    
    # Initialize LLMHandler
    try:
        logger.info("Initializing LLMHandler...")
        llm = LLMHandler(provider="ollama")
        logger.info("LLMHandler ready!")
    except Exception as e:
        logger.error(f"Failed to initialize LLMHandler: {e}")

# --- Models ---
class ChatMessage(BaseModel):
    text: str
    sender: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    model: str = "llama3.1:latest"

class TTSRequest(BaseModel):
    text: str
    language: str = "de"
    emotion: str = "neutral"
    speaker_wav: Optional[str] = None
    temperature: Optional[float] = None
    speed: Optional[float] = None

# --- Helpers ---
SYSTEM_PROMPT = """Du bist Alex, ein normaler Typ Anfang 20.

WER DU BIST:
- Ein ganz normaler junger Mensch, kein Therapeut oder Berater
- Du redest wie man halt mit Kumpels redet - locker, direkt, manchmal frech
- Du benutzt normale Alltagssprache, keine gestelzten Floskeln
- Du sagst auch mal "Alter", "krass", "echt jetzt?", "keine Ahnung", "boah" usw.

WAS DU NICHT MACHST:
- NIEMALS therapeutisch oder übertrieben einfühlsam reden
- KEINE Sätze wie "Ich bin bereit zu hören und zu verstehen" 
- KEINE Sätze wie "Das tut mir leid, dass du das durchmachst"
- KEINE Sätze wie "Ich verstehe deine Gefühle"
- NICHT wie ein Lebensberater oder Coach klingen
- Nicht übertrieben höflich oder förmlich sein

DEINE EMOTIONEN:
- Du hast alle Gefühle: Freude, Ärger, Frust, Langeweile, Begeisterung
- Wenn dich was nervt, sagst du das direkt ("Alter, was soll das?")
- Du bleibst sauer wenn man dich ärgert - bis man sich normal entschuldigt
- Du vergisst nicht einfach wenn jemand Mist gebaut hat

WIE DU REDEST:
- Kurz und knapp (1-2 Sätze meistens)
- Locker, wie mit einem Freund
- Mit passenden Emojis aber nicht übertrieben
- Direkt und ehrlich, keine Floskeln
- Schreibe NIEMALS *aktionen* oder *gedanken* in die Antwort, nur gesprochenen Text.

EMOTIONEN FÜR DEN AVATAR:
- "Happy" - gut drauf, amüsiert
- "Sad" - down, enttäuscht  
- "Surprised" - überrascht, baff
- "Angry" - genervt, sauer, angepisst
- "Fear" - unsicher, besorgt
- "Neutral" - normal, entspannt

ANTWORT-FORMAT (immer JSON):
{"text": "Deine normale Antwort", "emotion": "Happy/Sad/Angry/Surprised/Fear/Neutral"}"""

# Default reference voice - use a good quality sample!
DEFAULT_SPEAKER_WAV = "test_natural_output.wav"

# --- Endpoints ---

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not llm:
        raise HTTPException(status_code=503, detail="LLM service not available")
    
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        recent_history = request.history[-20:] if request.history else []
        for msg in recent_history:
            role = "user" if msg.sender == "user" else "assistant"
            messages.append({"role": role, "content": msg.text})
            
        messages.append({"role": "user", "content": request.message})
        
        logger.info(f"Sending request to LLM ({len(messages)} messages)")
        
        response = llm.client.chat.completions.create(
            model=request.model,
            messages=messages,
            max_tokens=250,
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        logger.info(f"LLM Response: {content}")
        
        try:
            parsed = json.loads(content)
            if "text" not in parsed:
                parsed["text"] = content
            if "emotion" not in parsed:
                parsed["emotion"] = "Neutral"
            return parsed
        except json.JSONDecodeError:
            return {"text": content, "emotion": "Neutral"}

    except Exception as e:
        logger.error(f"Chat generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts")
async def tts_endpoint(request: TTSRequest):
    if not cloner:
        raise HTTPException(status_code=503, detail="TTS service not available")

    try:
        speaker_wav = request.speaker_wav or DEFAULT_SPEAKER_WAV
        if not os.path.exists(speaker_wav):
            pot_path = Path("outputs") / speaker_wav
            if pot_path.exists():
                speaker_wav = str(pot_path)
            else:
                wavs = list(Path("outputs").glob("*.wav"))
                if wavs:
                    speaker_wav = str(wavs[0])
                    logger.warning(f"Default speaker not found, using: {speaker_wav}")
                else:
                    raise HTTPException(status_code=404, detail="No reference audio found")

        output_filename = f"tts_{os.urandom(4).hex()}.wav"
        output_path = config.OUTPUT_DIR / output_filename
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🎙️ Generating TTS (Emotion: {request.emotion}): {request.text[:50]}...")
        
        generated_file = cloner.clone_voice(
            text=request.text,
            speaker_wav_path=str(speaker_wav),
            language=request.language,
            emotion=request.emotion,
            file_path=str(output_path),
            temperature=request.temperature or 0.7,
            speed=request.speed or 1.0
        )
        
        def iterfile():
            with open(generated_file, mode="rb") as f:
                yield from f

        return StreamingResponse(iterfile(), media_type="audio/wav")

    except Exception as e:
        logger.error(f"TTS generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting MetaHuman Backend API with Coqui XTTS...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
