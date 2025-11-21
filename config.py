"""Configuration settings for AI Voice Cloner."""
import os
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# LLM Settings
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# TTS Settings
TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
TTS_DEVICE = "cpu"  # Force CPU for stability on macOS
DEFAULT_LANGUAGE = "de"

# Generation Settings
MAX_TOKENS = 500
TEMPERATURE = 0.7
# System prompts per language (used to instruct the LLM in the target language)
SYSTEM_PROMPTS = {
    "de": (
        "Du bist ein kreativer Assistent. "
        "Schreibe einen kurzen, natürlichen Absatz auf Deutsch "
        "basierend auf der Anfrage des Nutzers."
    ),
    "en": (
        "You are a creative assistant. "
        "Write a short, natural paragraph in English based on the user's request."
    ),
    "fr": (
        "Vous êtes un assistant créatif. "
        "Écrivez un court paragraphe naturel en français basé sur la demande de l'utilisateur."
    ),
    "es": (
        "Eres un asistente creativo. "
        "Escribe un breve párrafo natural en español basado en la solicitud del usuario."
    )
}

def get_system_prompt(language_code: str) -> str:
    """Return a language-specific system prompt. Falls back to German."""
    return SYSTEM_PROMPTS.get(language_code, SYSTEM_PROMPTS.get(DEFAULT_LANGUAGE))


# Tone / emotion instructions per language
TONE_INSTRUCTIONS = {
    "de": {
        "neutral": "Schreibe den Text in einem neutralen, sachlichen Ton.",
        "happy": "Schreibe den Text fröhlich und enthusiastisch.",
        "sad": "Schreibe den Text ruhig und nachdenklich / melancholisch.",
        "angry": "Schreibe den Text energisch und bestimmt.",
        "calm": "Schreibe den Text ruhig und gelassen.",
        "excited": "Schreibe den Text sehr begeistert und lebhaft."
    },
    "en": {
        "neutral": "Write the text in a neutral, factual tone.",
        "happy": "Write the text cheerful and enthusiastic.",
        "sad": "Write the text calm and reflective / melancholic.",
        "angry": "Write the text energetic and assertive.",
        "calm": "Write the text calm and composed.",
        "excited": "Write the text very excited and lively."
    },
    "fr": {
        "neutral": "Écrivez le texte dans un ton neutre et factuel.",
        "happy": "Écrivez le texte joyeux et enthousiaste.",
        "sad": "Écrivez le texte calme et réfléchi / mélancolique.",
        "angry": "Écrivez le texte énergique et affirmé.",
        "calm": "Écrivez le texte calme et serein.",
        "excited": "Écrivez le texte très enthousiaste et vivant."
    },
    "es": {
        "neutral": "Escribe el texto en un tono neutro y objetivo.",
        "happy": "Escribe el texto alegre y entusiasta.",
        "sad": "Escribe el texto tranquilo y reflexivo / melancólico.",
        "angry": "Escribe el texto enérgico y contundente.",
        "calm": "Escribe el texto con calma y serenidad.",
        "excited": "Escribe el texto muy entusiasmado y vivaz."
    }
}


def get_tone_instruction(language_code: str, emotion: str) -> str:
    """Return a short instruction for the LLM to write in a given emotion/tone."""
    lang = language_code if language_code in TONE_INSTRUCTIONS else DEFAULT_LANGUAGE
    emotion_key = emotion if emotion in TONE_INSTRUCTIONS.get(lang, {}) else "neutral"
    return TONE_INSTRUCTIONS[lang].get(emotion_key, TONE_INSTRUCTIONS[DEFAULT_LANGUAGE]["neutral"])

# Silence/post-processing settings (pydub)
# `SILENCE_THRESH_DB`: dBFS threshold under which audio is considered silence (negative value)
# `MIN_SILENCE_LEN_MS`: minimum length of silence to consider for removal (milliseconds)
# `KEEP_SILENCE_MS`: how much silence to keep when trimming (milliseconds)
SILENCE_THRESH_DB = -40
MIN_SILENCE_LEN_MS = 300
KEEP_SILENCE_MS = 80

# Advanced TTS generation parameters for more natural speech
# `TEMPERATURE`: controls randomness/variation (0.0-1.0, higher = more variation)
# `SPEED`: speech speed multiplier (0.5-2.0, 1.0 = normal)
# `REPETITION_PENALTY`: reduces repetitive patterns (1.0-2.0, higher = less repetition)
# `LENGTH_PENALTY`: affects sentence length preference (0.5-2.0)
TTS_TEMPERATURE = 0.75
TTS_SPEED = 1.0
TTS_REPETITION_PENALTY = 2.0
TTS_LENGTH_PENALTY = 1.0

# Logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
