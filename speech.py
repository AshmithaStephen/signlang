from io import BytesIO
from gtts import gTTS

def make_speech(text: str, language_code: str):
    try:
        audio = BytesIO()
        gTTS(text=text, lang=language_code, slow=False).write_to_fp(audio)
        audio.seek(0)
        return audio
    except Exception:
        return None
