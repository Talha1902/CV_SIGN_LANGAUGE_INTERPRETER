import numpy as np
import cv2
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import json

# Load gestures
gestures = os.listdir('gestures/')
images = []
labels = []
label_map = {gesture: idx for idx, gesture in enumerate(gestures)}

for gesture in gestures:
    gesture_path = os.path.join('gestures', gesture)
    for img_name in os.listdir(gesture_path):
        img = cv2.imread(os.path.join(gesture_path, img_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (50, 50))
        images.append(img)
        labels.append(label_map[gesture])

# Save the label map to a file
with open('label_map.json', 'w') as f:
    json.dump(label_map, f)

images = np.array(images).reshape(-1, 50, 50, 1)
labels = np.array(labels)

# Shuffle and split the data
images, labels = shuffle(images, labels)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes=len(gestures))
y_test = to_categorical(y_test, num_classes=len(gestures))

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(gestures), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the model
model.save('gesture_model.h5')
