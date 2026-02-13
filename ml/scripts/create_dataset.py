import os

import cv2
import mediapipe as mp
import numpy as np

from airmouse.gestures.registry import GESTURES as gestures
from airmouse.paths import DATASETS_DIR
from airmouse.vision.preprocess import preprocess


def save_hand_data(hand_data, gesture_id):
    # Store into a structured location to avoid huge flat directories.
    # data/raw/<class_id>/<shard>/<n>.npy
    gesture_path = os.path.join(DATASETS_DIR, "raw", str(gesture_id))
    if not os.path.exists(gesture_path):
        os.makedirs(gesture_path)

    # Shard by 1000 files per folder
    files = [
        f
        for f in os.listdir(gesture_path)
        if os.path.isfile(os.path.join(gesture_path, f)) and f.endswith(".npy")
    ]
    next_id = len(files) + 1
    shard = str((next_id - 1) // 1000).zfill(4)
    shard_dir = os.path.join(gesture_path, shard)
    os.makedirs(shard_dir, exist_ok=True)
    file_name = f"{next_id}.npy"
    file_path = os.path.join(shard_dir, file_name)

    np.save(file_path, np.array(hand_data))
    print(f"Data saved to {file_path}")


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

gesture_id = int(input(f"Gestures: {gestures}\nChoose the gesture number: "))

while True:
    hand_data = []
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_data = preprocess(hand_landmarks)

        cv2.imshow("Hand Tracking", image)
        if cv2.waitKey(5) & 0xFF == 32:
            save_hand_data(hand_data, gesture_id)
            break

cap.release()
cv2.destroyAllWindows()
