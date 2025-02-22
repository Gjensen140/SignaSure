from flask import Blueprint, request, jsonify
from signature_api.database import validate_pin
from signature_api.image_processor import image_cleaning

main = Blueprint("main", __name__)

@main.route('/clients/<client_id>/process_images', methods=['POST'])
def process_images(client_id):
    """Handles image uploads, converts them to NumPy arrays, and returns metadata"""
    if 'images' not in request.files:
        return jsonify({"error": "No images provided"}), 400

    images = request.files.getlist('images')
    processed_data = []

    for image in images:
        if image.filename == '':
            continue

        try:
            numpy_array = image_cleaning(image)
            processed_data.append({
                "filename": image.filename,
                "shape": numpy_array.shape,
                "dtype": str(numpy_array.dtype)
            })
        except Exception as e:
            return jsonify({"error": f"Failed to process image {image.filename}: {str(e)}"}), 500

    return jsonify({
        "message": f"Images processed successfully for client {client_id}!",
        "client_id": client_id,
        "processed_data": processed_data
    })
