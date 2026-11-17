import os
from google import genai

def get_ai_reply(user_text: str, language: str, history: list) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file.")
    client = genai.Client(api_key=api_key)
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    recent = history[-8:]
    transcript = "\n".join(
        f'{m["role"]}: {m["content"]}' for m in recent
    )
    prompt = f"""
You are a respectful, patient, accessibility-focused conversational assistant.
The user communicates through Indian Sign Language recognition. Recognition can contain errors:
ask a short clarifying question if the message is unclear; never pretend certainty.
Respond in {language}. Use clear, natural, concise language.
Conversation so far:
{transcript}

New user message:
{user_text}

Reply in {language}. Do not claim to be a certified sign-language interpreter.
"""
    response = client.models.generate_content(model=model, contents=prompt)
    if not response.text:
        raise RuntimeError("The AI returned an empty response.")
    return response.text.strip()
