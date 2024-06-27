import os
import cv2

def load_and_preprocess_images(gestures_dir):
    """
    Function to load and preprocess images from the specified directory.
    
    Args:
    - gestures_dir (str): Path to the directory containing gesture images.
    
    Returns:
    - images (list): List of preprocessed images (numpy arrays).
    """
    images = []
    labels = []
    label_map = {}
    gestures = os.listdir(gestures_dir)
    label_idx = 0

    for gesture in gestures:
        gesture_path = os.path.join(gestures_dir, gesture)
        label_map[gesture] = label_idx
        for img_name in os.listdir(gesture_path):
            img = cv2.imread(os.path.join(gesture_path, img_name), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (50, 50))
            images.append(img)
            labels.append(label_idx)
        label_idx += 1

    return images, labels, label_map

#Dispaying the images

if __name__ == "__main__":
    gestures_dir = 'gestures/'
    images = load_and_preprocess_images(gestures_dir)

    for img in images:
        cv2.imshow("Image", img)
        cv2.waitKey(30)

    cv2.destroyAllWindows()
