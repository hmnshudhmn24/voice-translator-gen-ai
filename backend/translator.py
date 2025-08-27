# backend/translator.py
import os
import uuid
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
TMP_DIR = Path(os.getenv("TMP_DIR", "./tmp"))
TMP_DIR.mkdir(parents=True, exist_ok=True)

USE_OPENAI_WHISPER = os.getenv("USE_OPENAI_WHISPER", "false").lower() in ("1", "true", "yes")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ASR: either OpenAI Whisper API or local whisper/faster-whisper
if USE_OPENAI_WHISPER:
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
    except Exception:
        openai = None
else:
    # Try faster-whisper then whisper
    try:
        from faster_whisper import WhisperModel
        _WHISPER_BACKEND = "faster-whisper"
        _model_instance = None
    except Exception:
        try:
            import whisper
            _WHISPER_BACKEND = "whisper"
            _model_instance = None
        except Exception:
            _WHISPER_BACKEND = None
            _model_instance = None

# Translation: MarianMT (Hindi -> English)
from transformers import MarianMTModel, MarianTokenizer
import torch
_TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-hi-en"
_trans_tok = MarianTokenizer.from_pretrained(_TRANSLATION_MODEL_NAME)
_trans_model = MarianMTModel.from_pretrained(_TRANSLATION_MODEL_NAME)

# TTS: Coqui TTS
try:
    from TTS.api import TTS
    _tts_model = None
except Exception:
    TTS = None
    _tts_model = None

def asr_local(audio_path: str) -> str:
    """Run local whisper/faster-whisper ASR and return transcribed text (language detected)."""
    global _model_instance, _WHISPER_BACKEND
    if _WHISPER_BACKEND == "faster-whisper":
        if _model_instance is None:
            # CPU model; change device="cuda" if GPU available
            _model_instance = WhisperModel("large-v2", device="cpu", compute_type="int8_float16")
        segments, info = _model_instance.transcribe(audio_path, beam_size=5)
        text = " ".join([seg.text for seg in segments])
        return text
    elif _WHISPER_BACKEND == "whisper":
        if _model_instance is None:
            _model_instance = whisper.load_model("large")
        res = _model_instance.transcribe(audio_path)
        return res.get("text", "")
    else:
        raise RuntimeError("No local whisper backend available. Install faster-whisper or whisper or set USE_OPENAI_WHISPER=true")

def asr_openai(audio_path: str) -> str:
    """Use OpenAI's Whisper transcription endpoint (requires OPENAI_API_KEY)."""
    try:
        import openai
    except Exception:
        raise RuntimeError("openai package not available")
    with open(audio_path, "rb") as f:
        # The exact client API may change; this is a basic example.
        resp = openai.Audio.transcriptions.create(file=f, model="whisper-1") if hasattr(openai, 'Audio') else openai.Audio.transcribe(file=f, model="whisper-1")
        if isinstance(resp, dict) and "text" in resp:
            return resp["text"]
        return str(resp)

def transcribe(audio_path: str) -> str:
    if USE_OPENAI_WHISPER:
        return asr_openai(audio_path)
    else:
        return asr_local(audio_path)

def translate_hi_to_en(text: str) -> str:
    if not text or text.strip() == "":
        return ""
    # Tokenize and generate
    inputs = _trans_tok([text], return_tensors="pt", padding=True)
    with torch.no_grad():
        translated = _trans_model.generate(**inputs, max_length=512)
    translated_text = _trans_tok.decode(translated[0], skip_special_tokens=True)
    return translated_text

def init_tts(model_name: Optional[str] = None):
    global _tts_model
    if TTS is None:
        raise RuntimeError("Coqui TTS library not installed or failed to import.")
    if _tts_model is None:
        model_name = model_name or os.getenv("TTS_MODEL", "")
        if model_name:
            _tts_model = TTS(model_name)
        else:
            # pick first available model
            models = TTS.list_models()
            if not models:
                raise RuntimeError("No TTS models available")
            _tts_model = TTS(models[0])
    return _tts_model

def synthesize_text_to_file(text: str, out_path: str, speaker_wav: Optional[str] = None):
    """Synthesize text and save WAV to out_path. If speaker_wav provided and model supports it,
    try to clone voice (depends on TTS model capabilities)."""
    tts = init_tts()
    kwargs = {}
    if speaker_wav:
        # Many Coqui multi-speaker models accept speaker_wav or speaker_idx; pass as available.
        kwargs["speaker_wav"] = speaker_wav
    # write to file
    tts.tts_to_file(text=text, file_path=out_path, **kwargs)
    return out_path

# Helper to ensure correct format (wav, mono, 16k)
from pydub import AudioSegment
def ensure_wav_mono_16k(in_path: str, out_path: str):
    seg = AudioSegment.from_file(in_path)
    seg = seg.set_channels(1)
    seg = seg.set_frame_rate(16000)
    seg.export(out_path, format="wav")
    return out_path

def pipeline_process(audio_file_path: str, speaker_sample_path: Optional[str] = None):
    """Full pipeline: ASR -> Translate -> TTS"""
    uid = uuid.uuid4().hex
    tmp_dir = TMP_DIR / uid
    tmp_dir.mkdir(parents=True, exist_ok=True)

    wav_in = str(tmp_dir / "input_16k.wav")
    ensure_wav_mono_16k(audio_file_path, wav_in)

    # transcribe
    try:
        transcribed = transcribe(wav_in)
    except Exception as e:
        transcribed = ""
        print("ASR error:", e)

    # translate
    try:
        translated = translate_hi_to_en(transcribed) if transcribed else ""
    except Exception as e:
        translated = ""
        print("Translation error:", e)

    # speaker sample processing
    speaker_wav = None
    if speaker_sample_path:
        speaker_wav = str(tmp_dir / "speaker_16k.wav")
        ensure_wav_mono_16k(speaker_sample_path, speaker_wav)

    # synthesize
    out_audio = str(tmp_dir / "translated.wav")
    try:
        synthesize_text_to_file(translated or " ", out_audio, speaker_wav=speaker_wav)
    except Exception as e:
        print("TTS error:", e)
        out_audio = ""

    return {
        "transcribed_text": transcribed,
        "translated_text": translated,
        "output_audio": out_audio,
    }
