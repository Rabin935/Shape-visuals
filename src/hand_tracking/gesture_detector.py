class GestureDetector:
    def count_fingers(self, landmarks):
        fingers = 0
        tips = [8, 12, 16, 20]

        for tip in tips:
            if landmarks[tip][1] < landmarks[tip - 2][1]:
                fingers += 1

        return fingers

    def detect_gesture(self, landmarks):
        count = self.count_fingers(landmarks)

        if count == 0:
            return "fist"
        if count == 1:
            return "point"
        if count == 2:
            return "peace"
        if count == 4:
            return "open"
        return "unknown"
