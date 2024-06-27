import cv2
import os

def is_image_file(filename):
    valid_extensions = ['.jpg', '.jpeg', '.png']
    return any(filename.lower().endswith(ext) for ext in valid_extensions)

# Get gesture name to display
gest_name = input("Enter gesture name to display: ")
path = os.path.join("gestures", gest_name)

if not os.path.exists(path):
    print(f"Error: The directory '{path}' does not exist.")
    exit()

images = [img for img in os.listdir(path) if is_image_file(img)]
if not images:
    print(f"No image files found in the directory '{path}'.")
    exit()

current_index = 0

while True:
    img_name = images[current_index]
    img_path = os.path.join(path, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Error: Unable to read '{img_name}'.")
        current_index += 1
        if current_index >= len(images):
            break
        continue

    cv2.imshow("Gesture", img)
    key = cv2.waitKey(0)

    if key == ord('q'):
        break
    elif key == ord('n'):
        current_index += 1
        if current_index >= len(images):
            current_index = 0
    elif key == ord('p'):
        current_index -= 1
        if current_index < 0:
            current_index = len(images) - 1

cv2.destroyAllWindows()
