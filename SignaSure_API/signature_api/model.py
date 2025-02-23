from flask import Flask, request, jsonify
import torch
import numpy as np

# Load the PyTorch model
model = torch.load("SignaSure_API\signature_api\\forge_2.h5", map_location=torch.device('cpu'),weights_only=False)
model.eval()

CONFIDENCE_THRESHOLD = 0.15  

app = Flask(__name__)

def classify(img1, img2):

    
    with torch.no_grad():
        prediction1 = model(img1).cpu().numpy().flatten()
        prediction2 = model(img2).cpu().numpy().flatten()
    
    similarity = np.dot(prediction1, prediction2) / (
        np.linalg.norm(prediction1) * np.linalg.norm(prediction2)
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


