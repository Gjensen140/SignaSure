from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model("SignaSure_API/signature_api/forge_2.h5")

CONFIDENCE_THRESHOLD = 0.15  # Adjust as needed
app = Flask(__name__)

def preprocess_image(img):
    """Ensure the image has the correct shape (1, 224, 224, 3) before passing to the model."""
    if img.shape != (224, 224, 3):
        raise ValueError(f"Invalid image shape {img.shape}, expected (224, 224, 3)")

    # Expand dimensions to simulate batch size (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)

    # Convert to TensorFlow tensor
    return tf.convert_to_tensor(img, dtype=tf.float32)

def classify(img1, img2):
    """Classifies the similarity between two images"""
    # Preprocess images
    img1_tensor = preprocess_image(img1)
    img2_tensor = preprocess_image(img2)

    print(f"Processed Image 1 Shape: {img1_tensor.shape}")  # Should be (1, 224, 224, 3)
    print(f"Processed Image 2 Shape: {img2_tensor.shape}")  # Should be (1, 224, 224, 3)

    # Make predictions
    prediction1 = model.predict(img1_tensor)
    prediction2 = model.predict(img2_tensor)

    # Compute similarity score
    prediction1_flat = prediction1.flatten()
    prediction2_flat = prediction2.flatten()

    epsilon = 1e-10  # Avoid division by zero
    similarity = np.dot(prediction1_flat, prediction2_flat) / (
        np.linalg.norm(prediction1_flat) * np.linalg.norm(prediction2_flat) + epsilon
    )

    confidence_score = float(abs(similarity))

    # Determine classification
    classification = "Forgery" if confidence_score > CONFIDENCE_THRESHOLD else "Genuine"

    return jsonify({
        'similarity_score': similarity,
        'confidence': confidence_score if classification == "Forgery" else (1 - confidence_score),
        'classification': classification
    })
