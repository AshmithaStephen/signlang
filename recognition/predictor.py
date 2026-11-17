from pathlib import Path
import json
import numpy as np

MODEL_PATH = Path("models/isl_lstm.keras")
LABELS_PATH = Path("models/labels.json")

def model_available():
    return MODEL_PATH.exists() and LABELS_PATH.exists()

def predict_sequence(sequence):
    if not model_available():
        return None, None
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("Install optional ML dependencies: pip install -r requirements-ml.txt") from exc

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    x = np.asarray(sequence, dtype=np.float32)
    if x.shape != (30, 226):
        raise ValueError(f"Expected (30, 226), got {x.shape}")
    probs = model.predict(x[None, ...], verbose=0)[0]
    idx = int(np.argmax(probs))
    return labels[idx], float(probs[idx])
