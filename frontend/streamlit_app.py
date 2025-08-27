# frontend/streamlit_app.py
import os
import requests
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Voice-to-Voice Translator", page_icon="🎙️", layout="centered")
st.title("🎙️ Voice-to-Voice Translator — Hindi → English")
st.write("Record or upload a short Hindi clip; receive translated English speech back (with optional voice cloning).")

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("🎧 Input")
    uploaded = st.file_uploader("Upload an audio file (wav/mp3/ogg) or record below", type=["wav","mp3","ogg"], accept_multiple_files=False)
    record = st.checkbox("📼 Use browser recording (experimental)")
    if record:
        st.info("Recording uses Streamlit file uploader fallback. For real recording, use browser extension or upload file.")

    speaker = st.file_uploader("Optional: upload speaker sample (wav) to clone voice", type=["wav"], key="spk")

with col2:
    st.subheader("⚙️ Options")
    src = st.selectbox("Source language", ["hi"], index=0)
    tgt = st.selectbox("Target language", ["en"], index=0)
    btn = st.button("🚀 Translate & Speak")

if btn:
    if not uploaded:
        st.error("Please upload an audio file.")
    else:
        with st.spinner("Sending audio to backend..."):
            files = {"audio": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            if speaker:
                files["speaker_sample"] = (speaker.name, speaker.getvalue(), speaker.type)
            data = {"src_lang": src, "tgt_lang": tgt}
            try:
                resp = requests.post(f"{API_URL}/translate", files=files, data=data, timeout=120)
            except Exception as e:
                st.error(f"Request failed: {e}")
                resp = None

        if resp and resp.status_code == 200:
            j = resp.json()
            st.success("✅ Translated!")
            st.markdown("**Transcribed (HI):**")
            st.write(j.get("transcribed_text",""))
            st.markdown("**Translated (EN):**")
            st.write(j.get("translated_text",""))

            audio_path = j.get("audio_file")
            if audio_path:
                # fetch audio
                try:
                    audio_resp = requests.get(f"{API_URL}/audio", params={"path": audio_path}, timeout=60)
                    if audio_resp.status_code == 200:
                        tmp = Path("tmp_client")
                        tmp.mkdir(exist_ok=True)
                        out_file = tmp / Path(audio_path).name
                        out_file.write_bytes(audio_resp.content)
                        st.audio(out_file, format="audio/wav")
                    else:
                        st.error("Could not fetch translated audio.")
                except Exception as e:
                    st.error(f"Failed to fetch audio: {e}")
        else:
            if resp is not None:
                st.error(f"Server error: {resp.status_code} — {resp.text}")
