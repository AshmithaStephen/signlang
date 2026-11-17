# ISL Connect — CodeSpark starter

This is a hackathon starter, not a certified or general-purpose ISL interpreter.

## Windows setup

Use Python 3.11 (64-bit). Open PowerShell in this repository:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks activation, run this once in that terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Add your Gemini API key to `.env`, then:
```powershell
streamlit run app.py
```

## Team ownership

- Member 1: `recognition/` — landmark extraction, phrase data collection, LSTM training.
- Member 2: `assistant.py` — Gemini prompts and dialogue context.
- Member 3: `app.py` — Streamlit interface and webcam user experience.
- Member 4: `auth.py`, `speech.py`, `schema.sql`, Docker/integration.

## Gemini

Create an API key in Google AI Studio: https://aistudio.google.com/app/apikey

Paste it into `.env`:
`GEMINI_API_KEY=your_key_here`

Never commit `.env` or paste API keys into GitHub.

## Collect phrase sequences (Member 1)

The collector records 30-frame sequences. Use the exact same label for the same sign/phrase.
Get an ISL-fluent person to validate every label and recording.

Install optional TensorFlow:
```powershell
pip install -r requirements-ml.txt
```

Collect at least 5–10 examples per phrase, ideally from multiple signers:
```powershell
python -m recognition.collect --label "I NEED HELP" --samples 10
python -m recognition.collect --label "MY NAME IS" --samples 10
python -m recognition.collect --label "THANK YOU" --samples 10
```

Train:
```powershell
python -m recognition.train_lstm
```

The app can then load `models/isl_lstm.keras` and `models/labels.json`.

**Evaluation warning:** with only a few samples, the model is a demo baseline. Keep some signers/videos out of training for a meaningful held-out test. Do not present training accuracy as test accuracy. A fixed 30-frame phrase classifier is not yet unrestricted continuous sentence recognition.

## Supabase (Member 4)

1. Create a Supabase project.
2. Copy Project URL and anon/public key into `.env`.
3. Open SQL Editor and run `schema.sql`.
4. Enable email/password authentication in Supabase Auth settings.

The app runs in demo mode without Supabase credentials. Use only the anon key in the app; never put a service-role key in a client app or GitHub.

## Git workflow

Each member creates a branch:
```powershell
git checkout -b feature/member1-recognition
git add .
git commit -m "Add recognition starter"
git push -u origin feature/member1-recognition
```

Use separate branch names for each member and merge through pull requests.


## Docker (Member 4 — final integration)

Build and run after the app works locally:
```powershell
docker build -t isl-connect .
docker run --env-file .env -p 8501:8501 isl-connect
```

Note: browser camera access through a locally hosted Streamlit page is easiest when opened at `http://localhost:8501`. A remote container deployment may require HTTPS and browser camera permissions; test that separately.
