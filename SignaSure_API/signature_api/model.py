from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model("SignaSure_API/signature_api/forge_2.h5")

CONFIDENCE_THRESHOLD = 0.15  # TODO: Adjust as needed

app = Flask(__name__)

def classify(img1, img2):
    
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