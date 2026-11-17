
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    try:
        import tensorflow as tf
    except ImportError:
        raise SystemExit("TensorFlow is not installed.")

    path = Path("data/sequences.npz")

    if not path.exists():
        raise SystemExit("No dataset found. Run the collector first.")

    data = np.load(path, allow_pickle=False)
    X = data["X"].astype("float32")
    raw_y = data["y"].astype(str)

    labels = sorted(set(raw_y.tolist()))

    if len(labels) < 2:
        raise SystemExit("Need at least 2 different sign labels.")

    counts = {
        label: int((raw_y == label).sum())
        for label in labels
    }

    # Each class needs at least 2 samples for a stratified split.
    if min(counts.values()) < 2:
        raise SystemExit(f"Not enough samples. Current counts: {counts}")

    print("Training dataset:", X.shape)
    print("Sign labels:", counts)

    label_to_id = {label: i for i, label in enumerate(labels)}
    y = np.asarray([label_to_id[v] for v in raw_y], dtype="int32")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(30, 226)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(len(labels), activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nStarting model training...")

    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=4,
        verbose=1
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    Path("models").mkdir(exist_ok=True)

    model.save("models/isl_lstm.keras")

    with open("models/labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print("\nTRAINING COMPLETED!")
    print(f"Held-out test accuracy: {accuracy:.3f}")
    print("Model saved: models/isl_lstm.keras")
    print("Labels saved: models/labels.json")
    print("WARNING: This small dataset is only a prototype.")


if __name__ == "__main__":
    main()
