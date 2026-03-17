from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision


MODEL_PATH = Path(__file__).with_name("models") / "hand_landmarker.task"


class HandTracker:
    def __init__(self):
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
        self.hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
        self.mp_draw = vision.drawing_utils

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.hand_landmarker.detect(mp_image)

        hands_data = []

        for hand_landmarks, handedness in zip(
            results.hand_landmarks,
            results.handedness,
        ):
            label = handedness[0].category_name if handedness else "Unknown"
            landmarks = [(landmark.x, landmark.y) for landmark in hand_landmarks]

            hands_data.append({
                "label": label,
                "landmarks": landmarks,
            })

            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.hand_connections,
            )

        return frame, hands_data
