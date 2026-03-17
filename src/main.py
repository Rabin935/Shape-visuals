import cv2
from hand_tracking.hand_tracker import HandTracker

cap = cv2.VideoCapture(0)
tracker = HandTracker()

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    frame, hands = tracker.process(frame)

    cv2.imshow("Gesture Visualizer", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()