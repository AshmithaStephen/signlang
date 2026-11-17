
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

st.title("Webcam Connection Test")

ctx = webrtc_streamer(
    key="camera-test-2",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=False,
)

st.write("Connection status:", ctx.state.playing)
