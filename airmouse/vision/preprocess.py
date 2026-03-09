import numpy as np


def preprocess(hand_landmarks) -> np.ndarray:
    def_x = hand_landmarks.landmark[0].x
    def_y = hand_landmarks.landmark[0].y
    def_z = hand_landmarks.landmark[0].z
    hand_points = np.array(
        [
            [landmark.x - def_x, landmark.y - def_y, landmark.z - def_z]
            for landmark in hand_landmarks.landmark
        ]
    )
    return np.asarray(hand_points).reshape(-1)
