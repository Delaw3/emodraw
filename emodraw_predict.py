import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os

# Load trained model
model = load_model("emodraw_model.h5")

# Labels from training folders (e.g., sun, tree, star)
labels = os.listdir("data")
labels.sort()

# Initialize MediaPipe hand tracker
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Drawing canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

draw = False  # Drawing mode
points = []

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lm = handLms.landmark[8]  # Index finger tip
            h, w, _ = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)

            if draw:
                points.append((cx, cy))
                for i in range(1, len(points)):
                    cv2.line(canvas, points[i - 1], points[i], (0, 255, 0), 4)

            cv2.circle(frame, (cx, cy), 8, (0, 0, 255), cv2.FILLED)
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    key = cv2.waitKey(1)

    if key == ord('d'):  # Toggle draw mode
        draw = not draw
        if not draw:
            # Make prediction
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64)) / 255.0
            input_img = resized.reshape(1, 64, 64, 1)
            cv2.imshow("Model Input", resized)

            prediction = model.predict(input_img)[0]
            predicted_index = np.argmax(prediction)
            predicted_label = labels[predicted_index]

            print("Prediction:", predicted_label)

            cv2.putText(frame, f"Prediction: {predicted_label}", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    if key == ord('c'):  # Clear canvas
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        points.clear()

    if key == ord('q'):
        break

    combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
    cv2.imshow("EmoDraw - Real-time Prediction", combined)

cap.release()
cv2.destroyAllWindows()
