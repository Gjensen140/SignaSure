import cv2
import numpy as np

def image_cleaning(image_file, target_size=(952, 1360)):
    """Converts an image to grayscale, resizes it, and returns a NumPy array"""
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img
