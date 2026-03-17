import cv2
from hand_tracking.hand_tracker import HandTracker
from hand_tracking.gesture_detector import GestureDetector

cap = cv2.VideoCapture(0)
tracker = HandTracker()
gesture_detector = GestureDetector()

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    frame, hands = tracker.process(frame)

    for hand in hands:
        gesture = gesture_detector.detect_gesture(hand["landmarks"])
        print(hand["label"], gesture)

    cv2.imshow("Gesture Visualizer", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
