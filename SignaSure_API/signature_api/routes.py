from flask import Blueprint, request, jsonify
from signature_api.database import validate_pin
from signature_api.model import classify
from signature_api.image_processor import image_cleaning

main = Blueprint("main", __name__)

@main.route('/clients/<client_id>/process_images', methods=['POST'])
def process_images(client_id):
    """Handles image uploads, converts them to NumPy arrays, and returns metadata"""
    if 'images' not in request.files:
        return jsonify({"error": "No images provided"}), 400

    # Get PIN from request headers
    user_pin = request.headers.get("PIN")
    
    # Checking the user has a PIN in the system
    if not user_pin or not validate_pin(user_pin):
        return jsonify({"error": "Invalid or missing PIN. Access denied."}), 403

    # Making sure the files sent were images
    if 'images' not in request.files:
        return jsonify({"error": "No images provided"}), 400

    images = request.files.getlist('images')
    comp_images = []

    for image in images:
        # Skipping blank files
        if image.filename == '':
            continue

        try:
            # Converting then storing the images for comparison
            numpy_array = image_cleaning(image)
            comp_images.append(numpy_array)
        except Exception as e:
            return jsonify({"error": f"Failed to process image {image.filename}: {str(e)}"}), 500

    return classify(comp_images[0], comp_images[1])
