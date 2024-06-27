import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json
import datetime

# Load the trained model
model = load_model('gesture_model.h5')

# Load the label map
with open('label_map.json', 'r') as f:
    label_map = json.load(f)
# Invert the label map to map indices back to gesture names
inv_label_map = {v: k for k, v in label_map.items()}

# Initialize camera
cam = cv2.VideoCapture(0)

# Open a text file to save predictions
with open('predicted_data.txt', 'w') as f:
    while True:
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)

        # Check if the frame is black
        if np.sum(frame) == 0:
            label = "black"
            confidence = 1.0
        else:
            cv2.rectangle(frame, (300, 300), (100, 100), (0, 255, 0), 2)
            roi = frame[100:300, 100:300]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(roi, (50, 50))
            roi = roi.reshape(1, 50, 50, 1)
            result = model.predict(roi)
            prediction = np.argmax(result)
            confidence = np.max(result)
            label = inv_label_map[prediction] if confidence > 0.5 else "NOTHING"

        # Display label and confidence
        display_text = f"{label} ({confidence*100:.2f}%)"
        cv2.putText(frame, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.putText(frame, "Press 'q' to Quit", (50, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame in full screen
        cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Frame", frame)

        # Save the prediction to the text file with a timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp}: {label} ({confidence*100:.2f}%)\n")
        f.flush()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()
