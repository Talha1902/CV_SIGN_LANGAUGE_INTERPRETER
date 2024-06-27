import cv2
import numpy as np
import os
import pickle

# Initialize the combined histogram
combined_hist = None

# Load gestures
gestures = os.listdir('gestures/')

for gesture in gestures:
    gesture_path = os.path.join('gestures', gesture)
    for img_name in os.listdir(gesture_path):
        img_path = os.path.join(gesture_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (50, 50))  # Resize image to a fixed size (if needed)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

        if combined_hist is None:
            combined_hist = hist
        else:
            combined_hist += hist

# Normalize the combined histogram
combined_hist = cv2.normalize(combined_hist, combined_hist).flatten()

# Save the combined histogram
with open("combined_hist.pkl", "wb") as f:
    pickle.dump(combined_hist, f)

print("Combined histogram created and saved as 'combined_hist.pkl'.")
