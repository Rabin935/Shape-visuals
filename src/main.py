import cv2
from hand_tracking.hand_tracker import HandTracker
from hand_tracking.gesture_detector import GestureDetector
from visuals.renderer_2d import Renderer2D

cap = cv2.VideoCapture(0)
tracker = HandTracker()
gesture_detector = GestureDetector()


renderer = Renderer2D()
position = (300, 300)





while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    frame, hands = tracker.process(frame)

    for hand in hands:
        if hand["label"] == "Left":
            lm = hand["landmarks"][8]
            position = (int(lm[0] * frame.shape[1]), int(lm[1] * frame.shape[0]))
        else:
            gesture = gesture_detector.detect_gesture(hand["landmarks"])
            renderer.update(gesture)

    frame = renderer.draw(frame, position)

    cv2.imshow("Gesture Visualizer", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
