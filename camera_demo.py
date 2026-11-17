
from collections import deque

import cv2
import mediapipe as mp

from recognition.landmarks import extract_keypoints
from recognition.predictor import predict_sequence

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    raise SystemExit

# Initialize MediaPipe
holistic = mp.solutions.holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0
)

# Keep the latest 30 frames
sequence = deque(maxlen=30)

print("Camera started!")
print("Press SPACE to recognize a sign.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam frame.")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    # Draw detected landmarks
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp.solutions.holistic.POSE_CONNECTIONS
        )

    if results.left_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp.solutions.holistic.HAND_CONNECTIONS
        )

    if results.right_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp.solutions.holistic.HAND_CONNECTIONS
        )

    # Extract features and store the latest 30 frames
    keypoints = extract_keypoints(results)
    sequence.append(keypoints)

    cv2.putText(
        frame,
        f"Frames: {len(sequence)}/30 | SPACE: Recognize | Q: Quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    cv2.imshow("ISL Recognition Demo", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord(" ") and len(sequence) == 30:
        try:
            label, confidence = predict_sequence(list(sequence))
            print(f"Prediction: {label} | Confidence: {confidence:.2%}")
        except Exception as e:
            print("Prediction error:", e)

cap.release()
holistic.close()
cv2.destroyAllWindows()

