import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os
import pyttsx3

# Initialize the speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speed if needed

# Load trained model
model = load_model("emodraw_model.h5")

labels = os.listdir("data")
labels.sort()

emoji_map = {
    'star': '⭐',
    'tree': '🌳',
    'sun': '☀️',
    'pot': '🍯',
    'flower': '🌸',
    'house': '🏠',
    'car': '🚗',
}

def overlay_icon(background, icon_path, x, y):
    if not os.path.exists(icon_path):
        print("❌ Icon not found:", icon_path)
        return background

    icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
    if icon is None:
        print("❌ Failed to load icon")
        return background

    icon = cv2.resize(icon, (100, 100))
    if y + 100 > background.shape[0] or x + 100 > background.shape[1]:
        return background

    if icon.shape[2] == 4:
        b, g, r, a = cv2.split(icon)
        mask = a / 255.0
        for c in range(3):
            background[y:y+100, x:x+100, c] = (
                background[y:y+100, x:x+100, c] * (1 - mask) + icon[:, :, c] * mask
            )
    else:
        background[y:y+100, x:x+100] = icon

    return background

# Initialize
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
mode = 'idle'  # 'draw', 'erase', 'idle'
points = []
last_icon_path = None

cap = cv2.VideoCapture(0)

prev_point = None
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lm = handLms.landmark[8]
            h, w, _ = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)

            

            if mode == 'draw':
                if prev_point is not None:
                    cv2.line(canvas, prev_point, (cx, cy), (0, 0, 255), 8)
                prev_point = (cx, cy)

            elif mode == 'erase':
                cv2.circle(canvas, (cx, cy), 20, (0, 0, 0), -1)
                prev_point = None
            else:
                prev_point = None

            color = (0, 255, 0) if mode == 'draw' else (0, 0, 255) if mode == 'erase' else (255, 255, 255)
            cv2.circle(frame, (cx, cy), 8, color, cv2.FILLED)

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    key = cv2.waitKey(1)

    if key == ord('d'):
        mode = 'draw' if mode != 'draw' else 'idle'
        print("Drawing Mode" if mode == 'draw' else "Drawing Off")

    elif key == ord('e'):
        mode = 'erase'
        print("🧽 Erase Mode")# inside loop 

    elif key == ord('c'):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        points.clear()
        print("Canvas Cleared")

    elif key == ord('q'):
        break

    elif key == ord('p'):
        # Make prediction
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64)) / 255.0
        input_img = resized.reshape(1, 64, 64, 1)
        cv2.imshow("Model Input", resized)

        prediction = model.predict(input_img)[0]
        predicted_index = np.argmax(prediction)
        predicted_label = labels[predicted_index]
        emoji = emoji_map.get(predicted_label, '❓')
        confidence = round(prediction[predicted_index] * 100, 2)

        print(f"Prediction: {predicted_label} ({confidence}%) {emoji}")
        cv2.putText(frame, f"{predicted_label} {emoji} ({confidence}%)",
                    (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 3)
        
        # 🗣️ Speak the result
        engine.say(f"You drew a {predicted_label}")
        engine.runAndWait()

        last_icon_path = f"icons/{predicted_label}.png"

    combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
    if last_icon_path:
        combined = overlay_icon(combined, last_icon_path, 520, 20)

    cv2.imshow("EmoDraw - Real-time Prediction", combined)

cap.release()
cv2.destroyAllWindows()
