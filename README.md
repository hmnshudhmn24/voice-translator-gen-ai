# 🎙️ Voice-to-Voice AI Translator (Hindi → English)

A near-real-time voice translator that converts spoken Hindi to spoken English using:
- **ASR**: Whisper (OpenAI API or local whisper/faster-whisper)
- **Translation**: MarianMT (`Helsinki-NLP/opus-mt-hi-en`)
- **TTS**: Coqui TTS (supports multi-speaker & some voice cloning modes)

This repo provides a backend (FastAPI) and a Streamlit frontend for quick experiments. It works best for short clips (1–20s). For continuous streaming, you'll need to add WebRTC streaming and streaming ASR/TTS endpoints.

---

## ✨ Features
- 🎤 Upload or record short audio clips (wav/mp3/ogg)
- 🈶 ASR (Hindi) → ✍️ text
- 🔁 Translate Hindi → English using a neural translation model
- 🔊 Synthesize English speech via Coqui TTS (optionally attempt voice cloning using a reference sample)
- ✅ Demo UI with Streamlit

---

## 🧰 Tech stack
- FastAPI (backend)
- Streamlit (frontend)
- Whisper / faster-whisper / OpenAI Whisper (ASR)
- Hugging Face Transformers (MarianMT) for translation
- Coqui TTS for speech generation
- Pydub for audio normalization

---

## 🚀 Quickstart (local, prototype)
> Recommended: use a conda env or venv. Some packages (TTS, torch) may require special installs (CUDA/non-CUDA).

1. Clone the repo
```bash
git clone https://github.com/yourname/voice-translator-gen-ai.git
cd voice-translator-gen-ai
```

2. Create venv & install
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Configure `.env`
```bash
cp .env.example .env
# Edit .env if using OpenAI Whisper or custom TTS model
```

4. Start backend
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

5. Start frontend (in another terminal)
```bash
streamlit run frontend/streamlit_app.py
```

6. Open the Streamlit URL and test by uploading a Hindi audio clip.

---

## 🛠️ Config options
- `USE_OPENAI_WHISPER=true` + `OPENAI_API_KEY` — use OpenAI speech-to-text (no local model download).
- If using local ASR (`faster-whisper` / `whisper`), the model weights will be downloaded automatically (large).
- `TTS_MODEL` env var: name of Coqui TTS model (e.g., `tts_models/en/ljspeech/tacotron2-DDC`). For voice cloning you need a model that supports multi-speaker or reference conditioning.

---

## 🔍 Limitations & Tips
- **Latency**: This prototype uses request/response for short clips. For "real" streaming, switch to WebRTC and streaming ASR/TTS.
- **Voice cloning**: Not all Coqui TTS models support cloning via a raw sample. You may need to compute speaker embeddings or choose a model explicitly labelled for cloning or multi-speaker synthesis.
- **Models & sizes**: Whisper "large" gives better ASR quality but requires lots of RAM. Faster-whisper and GPU are recommended for speed.
- **Translation accuracy**: MarianMT works well for straightforward sentences; for complicated academic text you might try a more advanced translation pipeline or an LLM.

---

## 🧩 Extensions (next steps)
- Replace translate step with an LLM (for contextual translation or disambiguation).
- Add streaming (WebRTC) frontend for true real-time conversation.
- Add speaker embedding extraction + higher-quality voice cloning pipeline.
- Run TTS on GPU for faster synthesis.

---

## ⚖️ License
MIT — use and extend freely.

---

## 🙌 Credits
- OpenAI Whisper (ASR) &/or whisper/faster-whisper
- Helsinki-NLP MarianMT models
- Coqui TTS
