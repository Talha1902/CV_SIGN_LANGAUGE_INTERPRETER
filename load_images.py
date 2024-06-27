import os
import cv2

# Load gestures
gestures = os.listdir('gestures/')

for gesture in gestures:
    gesture_path = os.path.join('gestures', gesture)
    for img_name in os.listdir(gesture_path):
        img = cv2.imread(os.path.join(gesture_path, img_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (50, 50))
        cv2.imshow("Image", img)
        cv2.waitKey(30)

cv2.destroyAllWindows()
