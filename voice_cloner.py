import os
import logging
import torch
from pathlib import Path
from TTS.api import TTS
import config
from pydub import AudioSegment, silence
import re
import tempfile
import uuid

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
        style_wav=None,
        crossfade_ms=40,
        temperature=None,
        speed=None,
        repetition_penalty=None,
        length_penalty=None,
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
            
            # Prepare kwargs for TTS
            tts_kwargs = {
                "text": text,
                "speaker_wav": speaker_wav_path,
                "language": language,
                "file_path": file_path,
            }

            # Attach emotion/style if provided
            if emotion:
                # we'll add emotion under several possible keys
                for k in ["emotion", "style", "style_name"]:
                    tts_kwargs[k] = emotion
            if style_wav:
                tts_kwargs["style_wav"] = style_wav
            
            # Add advanced generation parameters for more natural speech
            if temperature is not None:
                tts_kwargs["temperature"] = float(temperature)
            else:
                tts_kwargs["temperature"] = config.TTS_TEMPERATURE
            
            if speed is not None:
                tts_kwargs["speed"] = float(speed)
            else:
                tts_kwargs["speed"] = config.TTS_SPEED
            
            if repetition_penalty is not None:
                tts_kwargs["repetition_penalty"] = float(repetition_penalty)
            else:
                tts_kwargs["repetition_penalty"] = config.TTS_REPETITION_PENALTY
            
            if length_penalty is not None:
                tts_kwargs["length_penalty"] = float(length_penalty)
            else:
                tts_kwargs["length_penalty"] = config.TTS_LENGTH_PENALTY

            def _safe_tts_to_file(call_kwargs):
                """Try calling tts_to_file with retries and fallback for unsupported kwargs.

                The TTS backend may reject unknown model_kwargs (e.g. 'style' or 'style_name').
                This helper will attempt to parse the error message, remove offending keys,
                and retry with progressively smaller sets of kwargs.
                """
                # Make copies to avoid mutating the outer dict
                kw = dict(call_kwargs)

                # Attempt 1: full kwargs
                try:
                    self.tts.tts_to_file(**kw)
                    return
                except Exception as e1:
                    logger.warning(f"TTS first attempt failed: {e1}")
                    err_text = str(e1)

                # Attempt 2: if the error lists unused model_kwargs, try to remove them
                try:
                    m = re.search(r"The following `model_kwargs` are not used by the model:\s*\[([^\]]+)\]", err_text)
                    if m:
                        bad = [k.strip().strip("'\" ") for k in m.group(1).split(',')]
                        for k in bad:
                            if k in kw:
                                kw.pop(k, None)
                        logger.info(f"Removed unsupported model kwargs: {bad} and retrying")
                        self.tts.tts_to_file(**kw)
                        return
                except Exception as e2:
                    logger.warning(f"TTS second attempt failed after removing reported bad keys: {e2}")

                # Attempt 3: remove commonly optional keys and retry
                try:
                    for k in ["emotion", "style", "style_name", "style_wav", "temperature", "speed", "repetition_penalty", "length_penalty"]:
                        kw.pop(k, None)
                    logger.info("Removed optional emotion/style/generation keys and retrying")
                    self.tts.tts_to_file(**kw)
                    return
                except Exception as e3:
                    logger.warning(f"TTS third attempt failed after stripping optional keys: {e3}")

                # Attempt 4: minimal fallback
                try:
                    minimal = {k: kw[k] for k in ["text", "speaker_wav", "file_path", "language"] if k in kw}
                    logger.info("Trying minimal kwargs for synthesis")
                    self.tts.tts_to_file(**minimal)
                    return
                except Exception as e4:
                    logger.error(f"All TTS attempts failed: {e4}")
                    # raise the last exception so the outer logic can handle it
                    raise

            # If crossfade is requested, synthesize sentence-by-sentence and append with crossfade
            try:
                cross = int(crossfade_ms) if crossfade_ms is not None else 0
            except Exception:
                cross = 0

            if cross > 0:
                # Smart chunking: Group sentences into chunks of ~250 chars to preserve prosody
                # Split by sentence endings first (keeping punctuation)
                raw_sentences = [s.strip() for s in re.split(r'(?<=[\.\!\?])\s+', text) if s.strip()]
                
                chunks = []
                current_chunk = ""
                MAX_CHUNK_LENGTH = 250
                
                for sent in raw_sentences:
                    # If a single sentence is huge, we have to take it as is (or split further, but let's assume reasonable length)
                    if len(current_chunk) + len(sent) + 1 <= MAX_CHUNK_LENGTH:
                        current_chunk = (current_chunk + " " + sent).strip()
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sent
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If we only have one chunk effectively, just run it directly
                if len(chunks) <= 1:
                    _safe_tts_to_file(tts_kwargs)
                else:
                    logger.info(f"Split text into {len(chunks)} chunks for natural flow")
                    with tempfile.TemporaryDirectory() as td:
                        part_files = []
                        for i, chunk in enumerate(chunks):
                            part_path = Path(td) / f"part_{i}_{uuid.uuid4().hex}.wav"
                            part_kwargs = tts_kwargs.copy()
                            part_kwargs["text"] = chunk
                            part_kwargs["file_path"] = str(part_path)
                            try:
                                _safe_tts_to_file(part_kwargs)
                                part_files.append(str(part_path))
                            except Exception as e:
                                logger.warning(f"Failed to generate chunk {i}: {e}")
                                # If a chunk fails, we might skip it or fail. Let's fail to be safe.
                                raise

                        # concat with crossfade
                        combined = None
                        for p in part_files:
                            seg = AudioSegment.from_file(p)
                            if combined is None:
                                combined = seg
                            else:
                                combined = combined.append(seg, crossfade=cross)

                        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                        combined.export(file_path, format=Path(file_path).suffix.replace('.', ''))
            else:
                _safe_tts_to_file(tts_kwargs)

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
