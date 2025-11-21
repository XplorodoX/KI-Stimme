import gradio as gr
import os
import logging
from pathlib import Path
from voice_cloner import VoiceCloner
from pydub import AudioSegment
from llm_handler import LLMHandler
import config

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize modules
logger.info("Initializing AI Voice Cloner...")
try:
    cloner = VoiceCloner()
    logger.info("VoiceCloner initialized successfully")
except Exception as e:
    logger.error(f"Could not load VoiceCloner: {e}", exc_info=True)
    cloner = None

try:
    llm = LLMHandler(provider=config.LLM_PROVIDER)
    logger.info("LLMHandler initialized successfully")
except Exception as e:
    logger.error(f"Could not load LLMHandler: {e}", exc_info=True)
    llm = None

def process_pipeline(audio_file, prompt, language, emotion, trim_start=0.0, trim_end=0.0,
                     silence_thresh_db=None, min_silence_len_ms=None, keep_silence_ms=None,
                     style_ref=None, crossfade_ms=40, temperature=None, speed=None, 
                     repetition_penalty=None, length_penalty=None):
    """Process the complete pipeline: LLM text generation + voice cloning."""
    
    # Input validation
    if not audio_file:
        return "❌ Bitte laden Sie eine Audio-Datei hoch.", None
    
    if not prompt or len(prompt.strip()) == 0:
        return "❌ Bitte geben Sie einen Text-Prompt ein.", None
    
    if not cloner:
        return "❌ VoiceCloner konnte nicht initialisiert werden. Bitte prüfen Sie die Logs.", None
    
    if not llm:
        return "❌ LLM konnte nicht initialisiert werden. Bitte prüfen Sie die Logs.", None

    try:
        # 1. Generate Text from LLM
        logger.info(f"Generating text for prompt: {prompt[:50]}... (language={language})")
        # Build a language-specific system prompt including the requested tone/emotion
        base_prompt = config.get_system_prompt(language)
        tone_instruction = config.get_tone_instruction(language, emotion)
        system_prompt = f"{base_prompt} {tone_instruction}"

        generated_text = llm.generate_text(
            prompt,
            system_prompt=system_prompt
        )
        
        if generated_text.startswith("Error"):
            logger.error(f"LLM generation failed: {generated_text}")
            return f"❌ {generated_text}", None
        
        logger.info(f"Generated text ({len(generated_text)} chars): {generated_text[:100]}...")

        # 2. Clone Voice
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = config.OUTPUT_DIR / f"output_{timestamp}.wav"
        
        logger.info("Starting voice cloning...")
        # Prepare optional style reference path
        style_wav_path = str(style_ref) if style_ref else None

        cloner.clone_voice(
            generated_text,
            audio_file,
            language=language,
            emotion=emotion,
            file_path=str(output_path),
            silence_thresh_db=silence_thresh_db,
            min_silence_len_ms=min_silence_len_ms,
            keep_silence_ms=keep_silence_ms,
            style_wav=style_wav_path,
            crossfade_ms=int(crossfade_ms) if crossfade_ms is not None else 40,
            temperature=temperature,
            speed=speed,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
        )
        logger.info(f"Voice cloning completed: {output_path}")

        # Optional: trim the generated audio (front/back) using pydub
        try:
            trim_start = float(trim_start) if trim_start is not None else 0.0
            trim_end = float(trim_end) if trim_end is not None else 0.0
        except Exception:
            trim_start = 0.0
            trim_end = 0.0

        final_path = str(output_path)
        if trim_start > 0 or trim_end > 0:
            try:
                audio = AudioSegment.from_file(str(output_path))
                duration_ms = len(audio)
                start_ms = int(max(0, trim_start) * 1000)
                end_ms = int(max(0, duration_ms - int(trim_end * 1000)))
                if start_ms >= end_ms:
                    logger.warning("Trim parameters remove entire audio — skipping trimming")
                else:
                    trimmed = audio[start_ms:end_ms]
                    trimmed_path = str(output_path).replace('.wav', f'_trimmed_{int(start_ms)}_{int(end_ms)}.wav')
                    trimmed.export(trimmed_path, format='wav')
                    final_path = trimmed_path
                    logger.info(f"Trimmed audio saved to: {final_path}")
            except Exception as e:
                logger.error(f"Failed to trim audio: {e}", exc_info=True)

        return f"✅ Text generiert und Audio erstellt!\n\n{generated_text}", final_path
        
    except Exception as e:
        error_msg = f"Fehler in der Pipeline: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return f"❌ {error_msg}", None

# UI Layout
with gr.Blocks(title="AI Voice Cloner", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ AI Voice Cloner")
    gr.Markdown(
        "Laden Sie eine Stimm-Referenz hoch, geben Sie ein Thema ein, "
        "und lassen Sie die KI in dieser Stimme sprechen!"
    )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1️⃣ Stimm-Referenz")
            audio_input = gr.Audio(
                label="Referenz-Audio (hochladen oder aufnehmen)",
                type="filepath",
                sources=["upload", "microphone"]
            )
            gr.Markdown("*Tipp: 5-10 Sekunden klare Sprache sind optimal*")
            
            gr.Markdown("### 2️⃣ Text-Prompt")
            prompt_input = gr.Textbox(
                label="Thema / Prompt für die KI",
                placeholder="z.B. Erzähle eine kurze Geschichte über eine Katze im Weltraum.",
                lines=3
            )
            
            gr.Markdown("### 3️⃣ Ausgabe-Sprache")
            language_select = gr.Dropdown(
                label="Ausgabe-Sprache",
                choices=[
                    ("Deutsch", "de"),
                    ("English", "en"),
                    ("Français", "fr"),
                    ("Español", "es")
                ],
                value=config.DEFAULT_LANGUAGE
            )

            gr.Markdown("### 4️⃣ Emotion / Ton")
            emotion_select = gr.Dropdown(
                label="Emotion/Ton",
                choices=[
                    ("Neutral", "neutral"),
                    ("Fröhlich", "happy"),
                    ("Traurig", "sad"),
                    ("Wütend", "angry"),
                    ("Beruhigt", "calm"),
                    ("Aufgeregt", "excited")
                ],
                value="neutral"
            )

            gr.Markdown("### 5️⃣ Audio trimmen")
            trim_start = gr.Number(value=0.0, label="Sekunden zum Entfernen am Anfang", precision=2)
            trim_end = gr.Number(value=0.0, label="Sekunden zum Entfernen am Ende", precision=2)
            
            gr.Markdown("### 6️⃣ Silence Post-Processing")
            silence_thresh = gr.Slider(minimum=-60, maximum=-10, step=1, value=config.SILENCE_THRESH_DB, label="Silence Threshold (dB)")
            min_silence = gr.Slider(minimum=50, maximum=2000, step=50, value=config.MIN_SILENCE_LEN_MS, label="Min Silence Length (ms)")
            keep_sil = gr.Slider(minimum=0, maximum=500, step=10, value=config.KEEP_SILENCE_MS, label="Kept Silence (ms)")
            
            gr.Markdown("### 7️⃣ Stil-Referenz & Übergänge")
            style_ref = gr.Audio(label="Stil-Referenz (optional, kurz)", type="filepath", sources=["upload"]) 
            crossfade_ms = gr.Slider(minimum=0, maximum=500, step=5, value=40, label="Crossfade beim Zusammenfügen (ms)")
            
            gr.Markdown("### 8️⃣ Erweiterte Einstellungen (für natürlichere Sprache)")
            with gr.Row():
                temperature = gr.Slider(minimum=0.1, maximum=1.0, step=0.05, value=config.TTS_TEMPERATURE, label="Temperature (Variation)")
                speed = gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=config.TTS_SPEED, label="Geschwindigkeit")
            with gr.Row():
                repetition_penalty = gr.Slider(minimum=1.0, maximum=3.0, step=0.1, value=config.TTS_REPETITION_PENALTY, label="Repetition Penalty")
                length_penalty = gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=config.TTS_LENGTH_PENALTY, label="Length Penalty")
            
            generate_btn = gr.Button("🚀 Stimme generieren", variant="primary", size="lg")
        
        with gr.Column():
            gr.Markdown("### 📝 Generierter Text")
            text_output = gr.Textbox(
                label="KI-generierter Text",
                lines=8,
                max_lines=15
            )
            
            gr.Markdown("### 🔊 Geklontes Audio")
            audio_output = gr.Audio(label="Audio-Ausgabe")

    generate_btn.click(
        fn=process_pipeline,
        inputs=[
            audio_input,
            prompt_input,
            language_select,
            emotion_select,
            trim_start,
            trim_end,
            silence_thresh,
            min_silence,
            keep_sil,
            style_ref,
            crossfade_ms,
            temperature,
            speed,
            repetition_penalty,
            length_penalty,
        ],
        outputs=[text_output, audio_output]
    )
    
    gr.Markdown("---")
    gr.Markdown(
        "**Hinweis:** Die erste Generierung kann länger dauern, da die Modelle geladen werden müssen. "
        "Alle generierten Dateien werden im `outputs/` Ordner gespeichert."
    )

if __name__ == "__main__":
    demo.launch()
