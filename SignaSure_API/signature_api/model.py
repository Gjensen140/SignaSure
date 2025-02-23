from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import numpy as np

# Load model
class SiameseModel(torch.nn.Module):
    def __init__(self):
        super(SiameseModel, self).__init__()
        self.model = SigNet()
        self.probs = torch.nn.Linear(self.model.feature_space_size * 2, 1)
    
    def forward_once(self, img):
        return self.model.forward_once(img)
    
    def forward(self, img1, img2):
        img1 = img1.view(-1, 1, 150, 220).float().div(255)
        img2 = img2.view(-1, 1, 150, 220).float().div(255)
        embedding1 = self.forward_once(img1)
        embedding2 = self.forward_once(img2)
        output = torch.cat([embedding1, embedding2], dim=1)
        return embedding1, embedding2, self.probs(output)

class SigNet(torch.nn.Module):
    def __init__(self):
        super(SigNet, self).__init__()
        self.feature_space_size = 2048
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 96, 11, stride=4),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2),
            torch.nn.Conv2d(96, 256, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2),
            torch.nn.Conv2d(256, 384, 3, padding=1),
            torch.nn.ReLU()
        )
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(256 * 3 * 5, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU()
        )

    def forward_once(self, img):
        x = self.conv_layers(img)
        x = x.view(x.shape[0], -1)
        x = self.fc_layers(x)
        return x

# Initialize model and load weights
model = SiameseModel()
model.load_state_dict(torch.load("convnet_best_loss.pt", map_location=torch.device('cpu'))['model'])
model.eval()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((150, 220)),
    transforms.ToTensor()
])

app = Flask(__name__)

def classify(img_arr1, img_arr2):

    CONFIDENCE_THRESHOLD = 0.5  #TODO FIND THE RIGHT VALUE

    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Both image1 and image2 are required'}), 400
 
    image1 = np.frombuffer(img_arr1, dtype = np.unit8)
    image2 = np.frombuffer(img_arr2, dtype = np.unit8)
    
    img1 = transform(image1).unsqueeze(0)
    img2 = transform(image2).unsqueeze(0)
    
    with torch.no_grad():
        embedding1, embedding2, confidence = model(img1, img2)
        similarity = F.cosine_similarity(embedding1, embedding2).item()
        confidence_score = torch.sigmoid(confidence).item()
    
    if confidence_score > CONFIDENCE_THRESHOLD:
        return jsonify({
            'similarity_score': similarity,
            'confidence': confidence_score,
            'classification': 'Forgery'
        })
    else:
        return jsonify({
            'similarity_score': similarity,
            'confidence': 1 - confidence_score,
            'classification': 'Genuine'
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
