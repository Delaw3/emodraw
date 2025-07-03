import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# Define labels
labels = sorted(os.listdir("data"))  # e.g. ['car', 'flower', 'house', ...]
label_to_index = {name: idx for idx, name in enumerate(labels)}  # {'car': 0, 'flower': 1, ...}
IMG_SIZE = 64
X, y = [], []

# Load images from data folders
for label_name in labels:
    folder = os.path.join("data", label_name)
    for file in os.listdir(folder):
        if file.endswith(".png"):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) /255.0  # Normalize to [0, 1]
            X.append(img)
            y.append(label_to_index[label_name])


# Convert to numpy arrays
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) 
y = to_categorical(np.array(y), num_classes=len(labels))

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(labels), activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save model
model.save('emodraw_model.h5')
print("Model saved as emodraw_model.h5")
