
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

st.title("Webcam Connection Test")

ctx = webrtc_streamer(
    key="camera-test",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)

if ctx.state.playing:
    st.success("WebRTC connection established!")
else:
    st.info("Waiting for webcam connection...")
