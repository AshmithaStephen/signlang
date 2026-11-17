import argparse
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from recognition.landmarks import extract_keypoints

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help='Exact phrase label, e.g. "I NEED HELP"')
    parser.add_argument("--samples", type=int, default=15)
    args = parser.parse_args()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Close other apps using the camera.")

    holistic = mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0
    )
    sequences, current = [], []
    recording = False
    print("Press SPACE to record one 30-frame phrase sample. Press Q to quit.")
    while len(sequences) < args.samples:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        result = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if result.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, result.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS
            )
        if result.left_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, result.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS
            )
        if result.right_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, result.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS
            )

        if recording:
            current.append(extract_keypoints(result))
            cv2.putText(frame, f"RECORDING {len(current)}/30", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if len(current) >= 30:
                sequences.append(np.asarray(current[:30], dtype=np.float32))
                current, recording = [], False
                print(f"Saved sample {len(sequences)}/{args.samples} for {args.label}")
        else:
            cv2.putText(frame, f"{args.label} | SPACE = record | {len(sequences)}/{args.samples}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("ISL dataset collector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == 32 and not recording:
            current, recording = [], True

    cap.release()
    holistic.close()
    cv2.destroyAllWindows()

    if sequences:
        Path("data").mkdir(exist_ok=True)
        path = Path("data/sequences.npz")
        new_x = np.asarray(sequences, dtype=np.float32)
        new_y = np.asarray([args.label] * len(sequences))
        if path.exists():
            old = np.load(path, allow_pickle=False)
            new_x = np.concatenate([old["X"], new_x])
            new_y = np.concatenate([old["y"], new_y])
        np.savez_compressed(path, X=new_x, y=new_y)
        print(f"Dataset now has {len(new_y)} samples across {len(set(new_y))} labels.")
    else:
        print("No samples recorded.")

if __name__ == "__main__":
    main()
