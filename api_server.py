from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse, StreamingResponse, Response
import logging
import config
from llm_handler import LLMHandler
import os
import uvicorn
from pathlib import Path
from typing import List, Optional
import json
import edge_tts
import asyncio

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="MetaHuman Backend API", description="API for LLM Chat and TTS Generation")

# --- Edge TTS Configuration ---
# Florian is a natural, warm, friendly German male voice
EDGE_TTS_VOICE = "de-DE-FlorianMultilingualNeural"

# Emotion to voice rate/pitch mapping for more expressive speech
# Keep pitch values small to avoid Edge TTS errors
EMOTION_SETTINGS = {
    "neutral": {"rate": "+0%", "pitch": "+0Hz"},
    "happy": {"rate": "+5%", "pitch": "+10Hz"},
    "sad": {"rate": "-10%", "pitch": "-10Hz"},
    "angry": {"rate": "+10%", "pitch": "+5Hz"},
    "calm": {"rate": "-5%", "pitch": "-5Hz"},
    "excited": {"rate": "+10%", "pitch": "+10Hz"},
    "thinking": {"rate": "-5%", "pitch": "+0Hz"},
    "surprised": {"rate": "+5%", "pitch": "+10Hz"},
    "fear": {"rate": "+5%", "pitch": "+5Hz"},
}

# --- Global Instances ---
llm: Optional[LLMHandler] = None

# --- Initialization ---
@app.on_event("startup")
async def startup_event():
    global llm
    logger.info("Starting up backend services...")
    
    # Initialize LLMHandler
    try:
        logger.info("Initializing LLMHandler...")
        llm = LLMHandler(provider="ollama")
        logger.info("LLMHandler ready!")
    except Exception as e:
        logger.error(f"Failed to initialize LLMHandler: {e}")
    
    logger.info(f"🎙️ Using Edge TTS with voice: {EDGE_TTS_VOICE}")

# --- Models ---
class ChatMessage(BaseModel):
    text: str
    sender: str  # 'bot' or 'user'

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    model: str = "llama3.1:latest"

class TTSRequest(BaseModel):
    text: str
    language: str = "de"
    emotion: str = "neutral"
    voice: Optional[str] = None
    rate: Optional[str] = None
    pitch: Optional[str] = None

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

# --- Endpoints ---

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not llm:
        raise HTTPException(status_code=503, detail="LLM service not available")
    
    try:
        # Build messages list manually to support history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add history (limit to last 20 turns)
        recent_history = request.history[-20:] if request.history else []
        for msg in recent_history:
            role = "user" if msg.sender == "user" else "assistant"
            messages.append({"role": role, "content": msg.text})
            
        # Add current message
        messages.append({"role": "user", "content": request.message})
        
        logger.info(f"Sending request to LLM ({len(messages)} messages)")
        
        # Call Ollama via OpenAI client from llm_handler
        response = llm.client.chat.completions.create(
            model=request.model,
            messages=messages,
            max_tokens=250, # Keep it brief
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        logger.info(f"LLM Response: {content}")
        
        try:
            # Parse JSON from LLM
            parsed = json.loads(content)
            # Ensure keys exist
            if "text" not in parsed:
                parsed["text"] = content
            if "emotion" not in parsed:
                parsed["emotion"] = "Neutral"
            return parsed
        except json.JSONDecodeError:
            # Fallback if LLM forgets JSON
            return {"text": content, "emotion": "Neutral"}

    except Exception as e:
        logger.error(f"Chat generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts")
async def tts_endpoint(request: TTSRequest):
    """Generate natural sounding speech using Microsoft Edge TTS."""
    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Select voice (default: Florian - warm, friendly German male)
        voice = request.voice or EDGE_TTS_VOICE
        
        # Get emotion settings for pitch/rate adjustment
        emotion = request.emotion.lower() if request.emotion else "neutral"
        settings = EMOTION_SETTINGS.get(emotion, EMOTION_SETTINGS["neutral"])
        
        # Allow manual overrides
        rate = request.rate or settings["rate"]
        pitch = request.pitch or settings["pitch"]
        
        logger.info(f"🎙️ Edge TTS: voice={voice}, emotion={emotion}, rate={rate}, pitch={pitch}")
        logger.info(f"📝 Text: {text[:80]}...")
        
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
        
        logger.info(f"✅ Generated {len(audio_data)} bytes of audio")
        
        # Return audio directly as MP3 (Edge TTS outputs MP3)
        return Response(
            content=audio_data,
            media_type="audio/mpeg"
        )
        
    except Exception as e:
        logger.error(f"TTS generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/voices")
async def list_voices():
    """List available Edge TTS voices."""
    return {
        "current": EDGE_TTS_VOICE,
        "available": {
            "male": "de-DE-FlorianMultilingualNeural",
            "male_alt": "de-DE-ConradNeural",
            "female": "de-DE-SeraphinaMultilingualNeural",
            "female_alt": "de-DE-AmalaNeural",
        },
        "emotions": list(EMOTION_SETTINGS.keys())
    }

if __name__ == "__main__":
    print("🚀 Starting MetaHuman Backend API with Edge TTS...")
    print(f"🎙️ Voice: {EDGE_TTS_VOICE}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
