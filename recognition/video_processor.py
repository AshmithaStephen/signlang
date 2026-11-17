from collections import deque
import threading
import av
import cv2
import mediapipe as mp
from streamlit_webrtc import VideoProcessorBase
from recognition.landmarks import extract_keypoints

class HolisticProcessor(VideoProcessorBase):
    def __init__(self):
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0,
        )
        self.sequence = deque(maxlen=30)
        self.lock = threading.Lock()

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)

        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS
            )
        if results.left_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS
            )
        if results.right_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS
            )

        keypoints = extract_keypoints(results)
        with self.lock:
            self.sequence.append(keypoints)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def get_sequence(self):
        with self.lock:
            return list(self.sequence)
