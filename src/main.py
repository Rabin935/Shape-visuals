import cv2

from hand_tracking.gesture_detector import GestureDetector
from hand_tracking.hand_tracker import HandTracker
from visuals import renderer_3d


cap = cv2.VideoCapture(0)
tracker = HandTracker()
gesture_detector = GestureDetector()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame, hands = tracker.process(frame)

    right_hand_gesture = None

    for hand in hands:
        if hand["label"] == "Left":
            landmark = hand["landmarks"][8]
            position_x = int(landmark[0] * frame.shape[1])
            position_y = int(landmark[1] * frame.shape[0])
            renderer_3d.update_position(
                position_x,
                position_y,
                frame.shape[1],
                frame.shape[0],
            )
        else:
            right_hand_gesture = gesture_detector.detect_gesture(hand["landmarks"])

    if right_hand_gesture is not None:
        renderer_3d.update_transform(right_hand_gesture)
        cv2.putText(
            frame,
            f"Gesture: {right_hand_gesture}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    frame = renderer_3d.draw(frame)
    cv2.imshow("Gesture Visualizer", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
