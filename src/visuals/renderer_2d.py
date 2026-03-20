import cv2

class Renderer2D:
    def __init__(self):
        self.shape = "circle"
        self.color = (0, 255, 0)
        self.size = 30

    def update(self, gesture):
        if gesture == "fist":
            self.shape = "square"
        elif gesture == "peace":
            self.shape = "circle"
        elif gesture == "open":
            self.color = (230, 0, 0)




    def draw(self, frame, position):
        x, y = position

        if self.shape == "circle":
            cv2.circle(frame, (x, y), self.size, self.color, -1)

        elif self.shape == "square":
            cv2.rectangle(
                frame,
                (x - self.size, y - self.size),
                (x + self.size, y + self.size),
                self.color,
                -1
            )

        return frame