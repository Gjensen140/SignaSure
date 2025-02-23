from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model("forge_2.h5")


Image_Width, Image_Height = 224, 224
CONFIDENCE_THRESHOLD = 0.5  # TODO: Adjust as needed

def preprocess_image(img_arr):
    """Preprocesses a numpy image array for model inference."""
    img_arr = img_arr.astype('float32') / 255.0  
    img_arr = np.expand_dims(img_arr, axis=0) 
    return img_arr

app = Flask(__name__)

def classify(image_array_1, image_array_2):
    
    img1 = preprocess_image(image_array_1)
    img2 = preprocess_image(image_array_2)
    
    prediction1 = model.predict(img1)
    prediction2 = model.predict(img2)
    
    similarity = np.dot(prediction1.flatten(), prediction2.flatten()) / (
        np.linalg.norm(prediction1.flatten()) * np.linalg.norm(prediction2.flatten())
    )
    confidence_score = float(abs(similarity))
    
    if confidence_score > CONFIDENCE_THRESHOLD: 
        return jsonify({
            'similarity_score': similarity,
            'confidence': confidence_score,
            'classification': "Forgery"
        })
    else:
        return jsonify({
            'similarity_score': similarity,
            'confidence': 1 - confidence_score,
            'classification': "Genuine"
        })