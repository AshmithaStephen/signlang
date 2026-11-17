import os
import streamlit as st
from dotenv import load_dotenv
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from recognition.video_processor import HolisticProcessor
from recognition.predictor import predict_sequence, model_available
from assistant import get_ai_reply
from speech import make_speech
from auth import get_supabase, auth_panel

load_dotenv()
st.set_page_config(page_title="ISL Connect", page_icon="🤟", layout="wide")

st.title("🤟 ISL Connect")
st.caption("Continuous Indian Sign Language recognition and multilingual dialogue — hackathon MVP")

with st.sidebar:
    st.header("Account")
    user = auth_panel()
    st.divider()
    language = st.selectbox("Assistant response language", ["English", "Hindi", "Kannada"])
    lang_codes = {"English": "en", "Hindi": "hi", "Kannada": "kn"}

left, right = st.columns([1.15, 1])

with left:
    st.subheader("1. Sign using your webcam")
    st.info("Allow camera access. Sign one complete phrase at a time. The current prototype classifies fixed-length phrase sequences after a model has been trained.")
    ctx = webrtc_streamer(
    key="isl-webcam",
    mode=WebRtcMode.SENDONLY,
    video_processor_factory=HolisticProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },
    async_processing=True,
)
    if ctx.video_processor:
        seq = ctx.video_processor.get_sequence()
        st.caption(f"Landmark frames captured in current window: {len(seq)} / 30")
        if len(seq) == 30 and model_available():
            if st.button("Recognize current sign sequence", type="primary"):
                label, confidence = predict_sequence(seq)
                st.session_state["recognized_text"] = label or ""
                st.session_state["recognition_confidence"] = confidence
        elif not model_available():
            st.warning("Recognition model not trained yet. Member 1: collect labelled phrase sequences and run the training script.")
    else:
        st.caption("Start the webcam above to begin landmark tracking.")

with right:
    st.subheader("2. Recognized text")
    recognized = st.session_state.get("recognized_text", "")
    confidence = st.session_state.get("recognition_confidence")
    if recognized:
        st.success(recognized)
        if confidence is not None:
            st.metric("Model confidence", f"{confidence:.1%}")
    else:
        st.warning("No verified model prediction yet.")
    st.text_input(
        "Dialogue test input (fallback only — not a sign prediction)",
        key="typed_demo_text",
        placeholder="Example: I need help finding the hospital",
    )
    sign_text = recognized
    fallback_text = st.session_state.get("typed_demo_text", "").strip()
    use_fallback = st.checkbox("Use typed text to test AI dialogue while recognition model is being prepared")
    message = fallback_text if use_fallback else sign_text

    st.subheader("3. AI conversation")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for item in st.session_state.chat_history[-6:]:
        with st.chat_message(item["role"]):
            st.write(item["content"])

    if st.button("Send to AI assistant", disabled=not bool(message)):
        try:
            answer = get_ai_reply(message, language, st.session_state.chat_history)
            st.session_state.chat_history.append({"role": "user", "content": message})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.session_state["last_answer"] = answer

            sb = get_supabase()
            if sb and user:
                sb.table("conversation_history").insert({
                    "user_id": user["id"],
                    "recognized_text": message,
                    "assistant_response": answer,
                    "language": language,
                }).execute()
            st.rerun()
        except Exception as exc:
            st.error(f"AI request failed: {exc}")

    answer = st.session_state.get("last_answer")
    if answer:
        st.markdown("**Assistant response**")
        st.write(answer)
        audio = make_speech(answer, lang_codes[language])
        if audio:
            st.audio(audio, format="audio/mp3")
        else:
            st.caption("Speech unavailable. Check internet access and gTTS installation.")

st.divider()
st.caption("Research prototype. Phrase-level recognition is limited to the classes represented in the verified training dataset. Do not use it as a certified interpreter.")
