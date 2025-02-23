from PIL import Image
import numpy as np

def image_cleaning(image_file, target_size=(224, 224)):
    """Converts an image to RGB, resizes it, and returns a NumPy array with shape (224, 224, 3)."""
    # Convert the image to RGB (3 channels)
    img = image_file.convert("RGB")
   
    # Resize the image
    img = img.resize(target_size, Image.LANCZOS)
   
    # Convert the image to a NumPy array
    img_array = np.array(img, dtype=np.float32)
   
    # Normalize the pixel values to the range [0, 1]
    img_array /= 255.0

    # Ensure shape (224, 224, 3)
    # if img_array.shape != (224, 224, 3):
    #     raise ValueError(f"Unexpected shape: {img_array.shape}. Expected (224, 224, 3).")
   
    return img_array