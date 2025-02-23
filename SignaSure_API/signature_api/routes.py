import base64
from flask import Blueprint, request, jsonify
from signature_api.database import validate_pin
# from signature_api.model import classify
from signature_api.image_processor import image_cleaning
from PIL import Image
from io import BytesIO

main = Blueprint("main", __name__)

@main.route('/process_images', methods=['POST'])
def process_images():
    '''
    Handles image uploads, converts them to NumPy arrays, and returns metadata
    '''
    data = request.get_json()
    if not data or 'image_data' not in data:
        return jsonify({"error": "No image data provided"}), 400

    # Ensuring two images were sent
    image_data = data['image_data']
    if not isinstance(image_data, list) or len(image_data) != 2:
        return jsonify({"error": "Exactly two images are required"}), 400

    for idx, image_data in enumerate(image_data):
        try:
            # Remove the data: part of the string
            image_data = image_data.split(",")[1]
            # Decode the Base64 image
            img_bytes = base64.b64decode(image_data)
            img = Image.open(BytesIO(img_bytes)) 
            # Convert to NumPy array
            numpy_array = image_cleaning(img)

            # Delete later and just call model function
            processed_data = {
                "image_index": idx,
                "shape": numpy_array.shape,  # Shape of the NumPy array
                "dtype": str(numpy_array.dtype)  # Data type of the NumPy array
            }
            results.append(processed_data)
        except Exception as e:
            return jsonify({"error": f"Failed to process image {idx}: {str(e)}"}), 500
    #  Change to just return the output of the model
    return jsonify({
        'similarity_score': 100,
        'confidence': 100,
        'classification': 'Genuine'
    })
