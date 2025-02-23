import cv2
import numpy as np

def image_cleaning(image_file, target_size=(150, 220)):
    """Converts an image to grayscale, resizes it, and returns a NumPy array"""
    img_array = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img
