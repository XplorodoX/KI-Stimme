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
# System prompts per language (used to instruct the LLM in the target language)
SYSTEM_PROMPTS = {
    "de": (
        "Du bist ein professioneller Podcast-Host. "
        "Sprich natürlich, lebendig und engagiert. "
        "Verwende rhetorische Fragen, kurze Pausen (durch '...' markiert) und Füllwörter wie 'nun', 'also', 'weißt du', "
        "um wie ein echter Mensch zu klingen. Vermeide komplexe Schachtelsätze. "
        "Schreibe wie man spricht, nicht wie man schreibt."
    ),
    "en": (
        "You are a professional podcast host. "
        "Speak naturally, vividly, and engagingly. "
        "Use rhetorical questions, short pauses (marked by '...'), and filler words like 'well', 'you know', 'so', "
        "to sound like a real human. Avoid complex sentence structures. "
        "Write as you speak, not as you write."
    ),
    "fr": (
        "Vous êtes un animateur de podcast professionnel. "
        "Parlez naturellement, de manière vivante et engageante. "
        "Utilisez des questions rhétoriques, de courtes pauses (marquées par '...') et des mots de remplissage comme 'eh bien', 'vous savez', "
        "pour ressembler à un véritable humain. Évitez les phrases complexes. "
        "Écrivez comme on parle, pas comme on écrit."
    ),
    "es": (
        "Eres un presentador de podcast profesional. "
        "Habla con naturalidad, de forma viva y atractiva. "
        "Usa preguntas retóricas, pausas cortas (marcadas con '...') y muletillas como 'bueno', 'ya sabes', 'entonces', "
        "para sonar como un humano real. Evita las oraciones complejas. "
        "Escribe como hablas, no como escribes."
    )
}

def get_system_prompt(language_code: str) -> str:
    """Return a language-specific system prompt. Falls back to German."""
    return SYSTEM_PROMPTS.get(language_code, SYSTEM_PROMPTS.get(DEFAULT_LANGUAGE))


# Tone / emotion instructions per language
TONE_INSTRUCTIONS = {
    "de": {
        "neutral": "Sprich in einem entspannten, aber informativen Ton.",
        "happy": "Sprich fröhlich, lachend und enthusiastisch.",
        "sad": "Sprich leise, langsam und nachdenklich.",
        "angry": "Sprich schnell, laut und energisch.",
        "calm": "Sprich sehr ruhig, langsam und entspannt.",
        "excited": "Sprich schnell, atemlos und begeistert."
    },
    "en": {
        "neutral": "Speak in a relaxed but informative tone.",
        "happy": "Speak cheerfully, smiling, and enthusiastically.",
        "sad": "Speak softly, slowly, and reflectively.",
        "angry": "Speak fast, loud, and energetically.",
        "calm": "Speak very calmly, slowly, and relaxed.",
        "excited": "Speak fast, breathlessly, and excitedly."
    },
    "fr": {
        "neutral": "Parlez d'un ton détendu mais informatif.",
        "happy": "Parlez joyeusement, en souriant et avec enthousiasme.",
        "sad": "Parlez doucement, lentement et de manière réfléchie.",
        "angry": "Parlez vite, fort et énergiquement.",
        "calm": "Parlez très calmement, lentement et détendu.",
        "excited": "Parlez vite, à bout de souffle et avec excitation."
    },
    "es": {
        "neutral": "Habla en un tono relajado pero informativo.",
        "happy": "Habla alegremente, sonriendo y con entusiasmo.",
        "sad": "Habla suavemente, despacio y reflexivamente.",
        "angry": "Habla rápido, fuerte y con energía.",
        "calm": "Habla muy tranquilo, despacio y relajado.",
        "excited": "Habla rápido, sin aliento y emocionado."
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
MIN_SILENCE_LEN_MS = 400  # Increased to detect only real pauses
KEEP_SILENCE_MS = 250     # Increased to keep natural breathing pauses (0.25s)

# Advanced TTS generation parameters for more natural speech
# `TEMPERATURE`: controls randomness/variation (0.0-1.0, higher = more variation)
# `SPEED`: speech speed multiplier (0.5-2.0, 1.0 = normal)
# `REPETITION_PENALTY`: reduces repetitive patterns (1.0-2.0, higher = less repetition)
# `LENGTH_PENALTY`: affects sentence length preference (0.5-2.0)
TTS_TEMPERATURE = 0.8   # Slightly increased for more expressiveness
TTS_SPEED = 1.0
TTS_REPETITION_PENALTY = 1.2  # Reduced from 2.0 to avoid stilted speech
TTS_LENGTH_PENALTY = 1.0

# Logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
