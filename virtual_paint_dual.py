import cv2
import mediapipe as mp
import numpy as np
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Define colors (BGR)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (0, 0, 0)]
color_names = ["Blue", "Green", "Red", "Yellow", "Eraser"]
current_color = colors[0]
brush_thickness = 10
eraser_thickness = 50

# Drawing canvas
canvas = np.zeros((720, 640, 3), np.uint8)
prev_x, prev_y = 0, 0

# Draw color selection bar
def draw_color_palette(img):
    box_width = 128
    for i, (color, name) in enumerate(zip(colors, color_names)):
        x1 = i * box_width
        cv2.rectangle(img, (x1, 0), (x1 + box_width, 100), color, -1)
        cv2.putText(img, name, (x1 + 20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (255, 255, 255) if name != "Eraser" else (0, 0, 0), 3)

# Mediapipe Hands
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Left side (webcam feed)
        frame = cv2.resize(frame, (640, 720))
        draw_color_palette(frame)

        # Right side (drawing canvas)
        canvas_show = canvas.copy()

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_finger_tip = hand_landmarks.landmark[8]
            x, y = int(index_finger_tip.x * 640), int(index_finger_tip.y * 720)

            # If finger touches top palette
            if y < 100:
                box_index = x // 128
                if box_index < len(colors):
                    current_color = colors[box_index]
                    cv2.putText(frame, f"Selected: {color_names[box_index]}",
                                (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            else:
                # Drawing mode
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                if current_color == (0, 0, 0):  # Eraser mode
                    mask = np.zeros_like(canvas)
                    cv2.line(mask, (prev_x, prev_y), (x, y), (255, 255, 255), eraser_thickness)
                    canvas = cv2.bitwise_and(canvas, cv2.bitwise_not(mask))
                else:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), current_color, brush_thickness)

                prev_x, prev_y = x, y

            # Small indicator circle
            cv2.circle(frame, (x, y), 8, current_color, -1)
        else:
            prev_x, prev_y = 0, 0

        # Combine webcam and canvas
        combined = np.hstack((frame, canvas_show))

        cv2.imshow("Virtual Paint", combined)

        key = cv2.waitKey(1)
        if key == ord('s'):
            filename = f"drawing_{int(time.time())}.png"
            cv2.imwrite(filename, canvas)
            print(f"Drawing saved as {filename}")
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
