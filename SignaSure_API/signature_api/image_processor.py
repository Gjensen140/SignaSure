from PIL import Image
import numpy as np

def image_cleaning(image_file, target_size=(512, 512)):
    """Converts an image to grayscale, resizes it, and returns a NumPy array"""
    # Convert the image to grayscale
    img = image_file.convert("L")
    
    # Resize the image
    img = img.resize(target_size, Image.LANCZOS)
    
    # Convert the image to a NumPy array
    img_array = np.array(img, dtype=np.float32)
    
    # Normalize the pixel values to the range [0, 1]
    img_array /= 255.0
    
    return img_array