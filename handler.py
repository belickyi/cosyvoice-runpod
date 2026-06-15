#!/usr/bin/env python3
"""
RunPod Serverless Handler for CosyVoice3 TTS.

Uses FunAudioLLM/Fun-CosyVoice3-0.5B-2512 model for high-quality
multilingual text-to-speech with zero-shot voice cloning.

API:
    Input:
        - text: str (required) - Text to synthesize
        - prompt_wav: str (optional) - Base64 encoded reference audio for voice cloning
        - prompt_text: str (optional) - Transcript of reference audio
        - instruct_text: str (optional) - Instructions like language, emotion, speed
        - output_format: str (default: "wav")
        - stream: bool (default: false)

    Output:
        - audio: str - Base64 encoded WAV at 16000Hz
"""

import base64
import contextlib
import io
import logging
import os
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import numpy as np
import runpod
import torch
import torchaudio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add CosyVoice paths
COSYVOICE_DIR = Path("/app/CosyVoice")
sys.path.insert(0, str(COSYVOICE_DIR))
sys.path.insert(0, str(COSYVOICE_DIR / "third_party" / "Matcha-TTS"))

# Model configuration
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/pretrained_models/Fun-CosyVoice3-0.5B"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Audio settings
TARGET_SAMPLE_RATE = 16000  # Client expects 16kHz
MODEL_SAMPLE_RATE = 22050   # CosyVoice native rate (will be updated from model)

# Input limits
MAX_TEXT_LENGTH = 5000  # CosyVoice model limit
MAX_PROMPT_AUDIO_SIZE = 10 * 1024 * 1024  # 10 MB

# Default voice for when no prompt_wav is provided
DEFAULT_VOICE_PATH = Path("/app/default_voice.wav")

# Global model instance
cosyvoice = None


@contextlib.contextmanager
def temp_wav_file(audio_bytes: bytes) -> Generator[str, None, None]:
    """Context manager for temporary WAV file with guaranteed cleanup."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        yield temp_path
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def load_model():
    """Load CosyVoice model once at worker startup."""
    global cosyvoice, MODEL_SAMPLE_RATE

    if cosyvoice is not None:
        return

    logger.info(f"Loading CosyVoice model from: {MODEL_DIR}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    try:
        from cosyvoice.cli.cosyvoice import CosyVoice3

        # Load model (CosyVoice3 API changed - no longer accepts load_jit/load_trt/fp16)
        cosyvoice = CosyVoice3(str(MODEL_DIR))

        MODEL_SAMPLE_RATE = cosyvoice.sample_rate
        logger.info(f"Model loaded successfully")
        logger.info(f"Model sample rate: {MODEL_SAMPLE_RATE}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise


def decode_audio_base64(audio_b64: str) -> tuple:
    """
    Decode base64 audio to tensor.

    Returns:
        (audio_tensor, sample_rate)
    """
    audio_bytes = base64.b64decode(audio_b64)

    # Write to temp file for torchaudio to read
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name

    try:
        audio, sr = torchaudio.load(temp_path)
        return audio, sr
    finally:
        os.unlink(temp_path)


def resample_audio(audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample audio tensor."""
    if orig_sr == target_sr:
        return audio

    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    return resampler(audio)


def tensor_to_wav_base64(audio_tensor: torch.Tensor, sample_rate: int) -> str:
    """
    Convert audio tensor to base64 encoded WAV.

    Args:
        audio_tensor: Float tensor [channels, samples] or [samples]
        sample_rate: Sample rate

    Returns:
        Base64 encoded WAV string
    """
    # Ensure 2D tensor
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    # Resample to target sample rate
    if sample_rate != TARGET_SAMPLE_RATE:
        audio_tensor = resample_audio(audio_tensor, sample_rate, TARGET_SAMPLE_RATE)
        sample_rate = TARGET_SAMPLE_RATE

    # Convert to numpy
    audio_np = audio_tensor.squeeze().cpu().numpy()

    # Normalize and convert to int16
    if audio_np.max() > 1.0 or audio_np.min() < -1.0:
        audio_np = audio_np / max(abs(audio_np.max()), abs(audio_np.min()))

    audio_int16 = (audio_np * 32767).astype(np.int16)

    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    wav_bytes = wav_buffer.getvalue()
    return base64.b64encode(wav_bytes).decode('utf-8')


@torch.inference_mode()
def synthesize_tts(
    text: str,
    prompt_wav_b64: Optional[str] = None,
    prompt_text: Optional[str] = None,
    instruct_text: Optional[str] = None,
) -> torch.Tensor:
    """
    Synthesize text to speech.

    Args:
        text: Text to synthesize
        prompt_wav_b64: Base64 encoded reference audio for voice cloning
        prompt_text: Transcript of reference audio
        instruct_text: Instructions (language, emotion, speed)

    Returns:
        Audio tensor at MODEL_SAMPLE_RATE
    """
    audio_output = None

    # Decode prompt audio if provided
    prompt_audio_bytes = None
    if prompt_wav_b64:
        try:
            prompt_audio_bytes = base64.b64decode(prompt_wav_b64)
            logger.info(f"Using voice cloning with prompt audio ({len(prompt_audio_bytes)} bytes)")
        except Exception as e:
            logger.warning(f"Failed to decode prompt audio: {e}")
            prompt_audio_bytes = None

    # Helper to run inference with optional prompt file
    def run_inference(prompt_wav_path: Optional[str] = None):
        nonlocal audio_output

        if prompt_wav_path and prompt_text:
            # Zero-shot voice cloning
            logger.info("Mode: zero-shot voice cloning")

            # Format prompt text for CosyVoice3
            formatted_prompt = prompt_text
            if not prompt_text.startswith("You are"):
                formatted_prompt = f"You are a helpful assistant.<|endofprompt|>{prompt_text}"

            for output in cosyvoice.inference_zero_shot(
                tts_text=text,
                prompt_text=formatted_prompt,
                prompt_wav=prompt_wav_path,
                stream=False,
            ):
                audio_output = output['tts_speech']
                break  # Take first output

        elif prompt_wav_path:
            # Cross-lingual (no prompt_text needed)
            logger.info("Mode: cross-lingual")

            # Add language tag for Russian
            tts_text = text
            if not text.startswith("<|"):
                tts_text = f"<|ru|>{text}"

            for output in cosyvoice.inference_cross_lingual(
                tts_text=tts_text,
                prompt_wav=prompt_wav_path,
                stream=False,
            ):
                audio_output = output['tts_speech']
                break

        else:
            # No prompt audio provided - check for built-in speakers
            logger.info("Mode: checking for built-in speakers")

            # Try to get available speakers (method may be misspelled in CosyVoice)
            available_speakers = []
            for method_name in ['list_available_spks', 'list_avaliable_spks']:
                method = getattr(cosyvoice, method_name, None)
                if method and callable(method):
                    try:
                        available_speakers = method()
                        logger.info(f"Found speakers via {method_name}: {available_speakers}")
                        break
                    except Exception as e:
                        logger.warning(f"{method_name} failed: {e}")

            if available_speakers:
                # Use first available speaker with inference_sft
                spk_id = available_speakers[0]
                logger.info(f"Using built-in speaker: {spk_id}")

                for output in cosyvoice.inference_sft(
                    tts_text=text,
                    spk_id=spk_id,
                    stream=False,
                ):
                    audio_output = output['tts_speech']
                    break
            else:
                # No built-in speakers - use default voice from Docker image
                if DEFAULT_VOICE_PATH.exists():
                    logger.info(f"Using default voice: {DEFAULT_VOICE_PATH}")

                    # Use cross-lingual mode with default voice
                    tts_text = text
                    if not text.startswith("<|"):
                        tts_text = f"<|ru|>{text}"

                    for output in cosyvoice.inference_cross_lingual(
                        tts_text=tts_text,
                        prompt_wav=str(DEFAULT_VOICE_PATH),
                        stream=False,
                    ):
                        audio_output = output['tts_speech']
                        break
                else:
                    logger.error("No built-in speakers and no default voice available.")
                    raise ValueError(
                        "No default voice configured. "
                        "Please provide 'prompt_wav' (base64 encoded WAV) for voice cloning."
                    )

    # Run inference with proper temp file cleanup
    if prompt_audio_bytes:
        with temp_wav_file(prompt_audio_bytes) as prompt_wav_path:
            run_inference(prompt_wav_path)
    else:
        run_inference(None)

    if audio_output is None:
        raise RuntimeError("No audio output generated")

    # Move to CPU to free GPU memory
    audio_output = audio_output.cpu()

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return audio_output


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler.

    Expected input schema:
    {
        "input": {
            "text": str (required),
            "prompt_wav": str (optional, base64 WAV),
            "prompt_text": str (optional),
            "instruct_text": str (optional),
            "output_format": str (default: "wav")
        }
    }

    Output schema:
    {
        "output": {
            "audio": str (base64 WAV at 16000Hz)
        }
    }
    """
    try:
        job_input = job.get("input", {})

        # Validate required fields
        text = job_input.get("text")
        if not text:
            return {"error": "Missing required field: text"}

        if len(text) > MAX_TEXT_LENGTH:
            return {"error": f"Text too long (max {MAX_TEXT_LENGTH} characters)"}

        # Validate prompt audio size
        prompt_wav = job_input.get("prompt_wav")
        if prompt_wav:
            # Estimate decoded size (base64 is ~1.33x larger than binary)
            estimated_size = len(prompt_wav) * 3 / 4
            if estimated_size > MAX_PROMPT_AUDIO_SIZE:
                return {"error": f"Prompt audio too large (max {MAX_PROMPT_AUDIO_SIZE / 1024 / 1024:.1f} MB)"}

        # Extract optional parameters (prompt_wav already validated above)
        prompt_text = job_input.get("prompt_text")
        instruct_text = job_input.get("instruct_text")

        logger.info(f"Synthesizing: {len(text)} chars")

        # Synthesize audio
        audio_tensor = synthesize_tts(
            text=text,
            prompt_wav_b64=prompt_wav,
            prompt_text=prompt_text,
            instruct_text=instruct_text,
        )

        # Convert to base64 WAV at 16kHz
        audio_b64 = tensor_to_wav_base64(audio_tensor, MODEL_SAMPLE_RATE)

        logger.info(f"Synthesis complete: {len(audio_tensor)} samples")

        return {
            "output": {
                "audio": audio_b64
            }
        }

    except Exception as e:
        logger.error(f"Handler error: {e}", exc_info=True)
        return {"error": str(e)}


# RunPod entry point
if __name__ == "__main__":
    logger.info("Starting CosyVoice3 TTS handler...")

    # Load model at startup
    load_model()

    # Start RunPod serverless handler
    runpod.serverless.start({"handler": handler})
