# backend/main.py
import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from backend.translator import pipeline_process

TMP_DIR = Path(os.getenv("TMP_DIR", "./tmp"))
TMP_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Voice-to-Voice Translator API")

@app.post("/translate")
async def translate_endpoint(
    audio: UploadFile = File(...),
    speaker_sample: UploadFile = File(None),
    src_lang: str = Form("hi"),
    tgt_lang: str = Form("en"),
):
    """
    Accepts:
      - audio: user speech (wav/mp3/ogg)
      - speaker_sample: optional sample wav to clone voice
    Returns: JSON with transcribed and translated text and audio file path.
    """
    uid = os.urandom(8).hex()
    base = TMP_DIR / uid
    base.mkdir(parents=True, exist_ok=True)
    audio_path = base / f"input_{audio.filename}"
    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    speaker_path = None
    if speaker_sample:
        spath = base / f"speaker_{speaker_sample.filename}"
        with open(spath, "wb") as f:
            f.write(await speaker_sample.read())
        speaker_path = str(spath)

    result = pipeline_process(str(audio_path), speaker_sample_path=speaker_path)
    if result.get("output_audio"):
        return {
            "transcribed_text": result.get("transcribed_text", ""),
            "translated_text": result.get("translated_text", ""),
            "audio_file": str(result.get("output_audio")),
        }
    else:
        return JSONResponse(status_code=500, content={"error": "TTS failed or missing output audio", "transcribed": result.get("transcribed_text"), "translated": result.get("translated_text")})

@app.get("/audio")
def get_audio(path: str):
    p = Path(path)
    if p.exists():
        return FileResponse(p, media_type="audio/wav", filename=p.name)
    return JSONResponse(status_code=404, content={"error": "file not found"})
