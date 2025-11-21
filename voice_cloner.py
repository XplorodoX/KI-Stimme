import os
import logging
import torch
from pathlib import Path
from TTS.api import TTS
import config
from pydub import AudioSegment, silence

logger = logging.getLogger(__name__)

class VoiceCloner:
    """Voice cloning using Coqui TTS XTTS model."""
    
    def __init__(self, model_name=None, use_gpu=False):
        """
        Initialize Voice Cloner.
        
        Args:
            model_name: TTS model to use
            use_gpu: Whether to use GPU (currently forced to CPU for stability)
        """
        # Auto-accept Coqui TOS for XTTS
        os.environ["COQUI_TOS_AGREED"] = "1"
        
        model_name = model_name or config.TTS_MODEL
        self.device = config.TTS_DEVICE
        logger.info(f"Loading TTS model '{model_name}' on {self.device}...")
        
        try:
            self.tts = TTS(model_name).to(self.device)
            logger.info("TTS model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}", exc_info=True)
            raise

    def clone_voice(
        self,
        text,
        speaker_wav_path,
        language="de",
        emotion=None,
        file_path="output.wav",
        silence_thresh_db=None,
        min_silence_len_ms=None,
        keep_silence_ms=None,
    ):
        """
        Generate audio from text using the speaker's voice.
        
        Args:
            text: Text to synthesize
            speaker_wav_path: Path to reference audio file
            language: Language code (e.g., 'de', 'en')
            file_path: Output file path
            
        Returns:
            Path to generated audio file
        """
        # Validate inputs
        if not text or len(text.strip()) == 0:
            raise ValueError("Text cannot be empty")
        
        if not os.path.exists(speaker_wav_path):
            raise FileNotFoundError(f"Reference audio file not found: {speaker_wav_path}")
        
        # Check file size (basic validation)
        file_size = os.path.getsize(speaker_wav_path)
        if file_size < 1000:  # Less than 1KB
            raise ValueError(f"Reference audio file seems too small: {file_size} bytes")
        
        logger.info(f"Cloning voice: text_length={len(text)}, language={language}")
        
        try:
            # Ensure output directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Try to pass emotion/style to the underlying TTS if supported.
            # The Coqui TTS `tts_to_file` signature may accept extra kwargs like `style_wav` or `emotion`.
            tts_kwargs = {
                "text": text,
                "speaker_wav": speaker_wav_path,
                "language": language,
                "file_path": file_path
            }

            if emotion:
                # We try common parameter names; if the TTS implementation does not accept them,
                # we'll fall back to calling without emotion.
                possible_emotion_keys = ["emotion", "style", "style_name", "style_wav"]
                for k in possible_emotion_keys:
                    tts_kwargs[k] = emotion

            try:
                self.tts.tts_to_file(**tts_kwargs)
            except TypeError:
                # Fallback: call without emotion-related keys
                for k in ["emotion", "style", "style_name", "style_wav"]:
                    tts_kwargs.pop(k, None)
                logger.warning("TTS backend did not accept emotion/style kwargs; falling back to default synthesis")
                self.tts.tts_to_file(**tts_kwargs)

            # Post-process: remove or reduce long silences between sentences
            try:
                # Load generated audio
                audio = AudioSegment.from_file(file_path)
                # detect silent ranges longer than configured minimum
                # Use provided parameters when available, otherwise fall back to config
                s_thresh = silence_thresh_db if silence_thresh_db is not None else config.SILENCE_THRESH_DB
                min_sil = min_silence_len_ms if min_silence_len_ms is not None else config.MIN_SILENCE_LEN_MS
                keep_ms = keep_silence_ms if keep_silence_ms is not None else config.KEEP_SILENCE_MS

                silent_ranges = silence.detect_silence(
                    audio,
                    min_silence_len=min_sil,
                    silence_thresh=s_thresh,
                )

                if silent_ranges:
                    # Build new audio by keeping up to KEEP_SILENCE_MS of each silence
                    parts = []
                    prev_end = 0
                    keep = keep_ms
                    for (start_ms, end_ms) in silent_ranges:
                        # append non-silent segment before this silence
                        if start_ms > prev_end:
                            parts.append(audio[prev_end:start_ms])
                        # append a short silence slice (keep)
                        parts.append(AudioSegment.silent(duration=keep))
                        prev_end = end_ms
                    # append the trailing non-silent audio
                    if prev_end < len(audio):
                        parts.append(audio[prev_end:])

                    processed = sum(parts)
                    # overwrite output file (or save as trimmed file)
                    processed.export(file_path, format=Path(file_path).suffix.replace('.', ''))
                    logger.info(f"Post-processed audio to reduce long silences: {file_path}")
                else:
                    logger.debug("No long silence ranges detected; skipping post-processing")
            except Exception as e:
                logger.warning(f"Could not post-process silences: {e}", exc_info=True)
            
            logger.info(f"Voice cloning successful: {file_path}")
            return file_path
            
        except Exception as e:
            error_msg = f"Failed to clone voice: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

if __name__ == "__main__":
    # Simple test
    cloner = VoiceCloner()
    # Create a dummy file for testing if needed, or expect one
    # cloner.clone_voice("Hallo, das ist ein Test.", "path/to/sample.wav")
