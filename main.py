import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

# Create folders
labels = {
    ord('1'): 'house', 
    ord('2'): 'star', 
    ord('3'): 'sun', 
    ord('4'): 'tree', 
    ord('5'): 'flower', 
    ord('6'): 'car'
    } #Data folder created here

for label in labels.values():
    os.makedirs(f'data/{label}', exist_ok=True)

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
h, w, _ = frame.shape
canvas = np.zeros((h, w, 3), dtype=np.uint8)

prev_x, prev_y = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            if prev_x != 0 and prev_y != 0:
                cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 255), 10) #Color for pen

            prev_x, prev_y = x, y
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        prev_x, prev_y = 0, 0

    combined = cv2.add(frame, canvas)
    cv2.imshow("EmoDraw - Drawing + Save", combined)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
    elif key in labels:
        label = labels[key]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        img_name = f"data/{label}/{label}_{timestamp}.png"
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        cv2.imwrite(img_name, resized)
        print(f"[✓] Saved: {img_name}")

cap.release()
cv2.destroyAllWindows()
