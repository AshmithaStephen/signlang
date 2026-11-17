
import numpy as np

# 2 hands * 21 landmarks * 3 coordinates = 126
# 25 pose landmarks * 4 values = 100
# Total = 226 features per frame

FEATURES = 226


def extract_keypoints(results):
    values = []

    # Extract left and right hand landmarks
    for hand in (
        results.left_hand_landmarks,
        results.right_hand_landmarks
    ):
        if hand:
            for point in hand.landmark:
                values.extend([point.x, point.y, point.z])
        else:
            values.extend([0.0] * (21 * 3))

    # Extract upper-body pose landmarks
    if results.pose_landmarks:
        pose = results.pose_landmarks.landmark[:25]

        for point in pose:
            values.extend([
                point.x,
                point.y,
                point.z,
                point.visibility
            ])

        # Fill any missing pose landmarks
        if len(pose) < 25:
            values.extend([0.0] * ((25 - len(pose)) * 4))

    else:
        values.extend([0.0] * (25 * 4))

    # Convert to a flat NumPy array
    arr = np.asarray(values, dtype=np.float32)

    if arr.size != FEATURES:
        raise ValueError(
            f"Expected {FEATURES} features, got {arr.size}"
        )

    return arr
