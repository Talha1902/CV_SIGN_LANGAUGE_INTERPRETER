import cv2
import os

def create_directory(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory: {e}")
        exit()

def initialize_camera():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open webcam.")
        exit()
    return cam

def capture_gesture_images():
    # Get gesture name
    gest_name = input("Enter gesture name: ")
    path = os.path.join("gestures", gest_name)
    create_directory(path)

    # Initialize camera
    cam = initialize_camera()

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (300, 400), (100, 100), (0, 255, 0), 2)
        cv2.imshow("Gesture Capture", frame)
        img_name = os.path.join(path, "{}.png".format(img_counter))

        key = cv2.waitKey(1)
        if key == ord('c'):
            try:
                roi = frame[100:400, 100:300]
                for angle in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                    rotated = cv2.rotate(roi, angle)
                    cv2.imwrite(img_name.replace('.png', '_{}.png'.format(angle)), rotated)
                    img_counter += 1
            except Exception as e:
                print(f"Error saving image: {e}")
        elif key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

capture_gesture_images()
