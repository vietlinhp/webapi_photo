from flask import Flask, request, jsonify
import cv2
import numpy as np
from io import BytesIO
import json
import requests
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Helper functions
def quality_check(image):
    params = {
        'models': 'quality',
        'api_user': '551714069',  # Replace with your API user
        'api_secret': 'JM2H5dexveresM2eaFQkLAAY3p6TUpna'  # Replace with your API secret
    }
    files = {'media': image}
    r = requests.post('https://api.sightengine.com/1.0/check.json', files=files, data=params)
    output = json.loads(r.text)
    return output["quality"]["score"]

def isbright(image_data, dim=10, thresh=0.5):
    image_array = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    image_resized = cv2.resize(image_array, (dim, dim))
    L, A, B = cv2.split(cv2.cvtColor(image_resized, cv2.COLOR_BGR2LAB))
    L = L / np.max(L)
    return np.mean(L) > thresh

def similarity_check(image_data_1, image_data_2):
    img1 = cv2.imdecode(np.frombuffer(image_data_1, np.uint8), 0)
    img2 = cv2.imdecode(np.frombuffer(image_data_2, np.uint8), 0)
    if img1 is None or img2 is None:
        raise ValueError(f"One or both images could not be loaded. Check the inputs.")
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        raise ValueError("Descriptors could not be found in one or both images.")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    similarity_score = len(matches) / min(len(kp1), len(kp2))
    return similarity_score

def photo_quality(image_data, image_data_comp):
    if quality_check(BytesIO(image_data)) < 0.55:
        return "Bad quality"
    if not isbright(image_data):
        return "The picture is too dark"
    if similarity_check(image_data, image_data_comp) < 0.30:
        return "The picture doesn't capture the object correctly"
    return "Great picture"

# Flask app definition
app = Flask(__name__)

@app.route('/photo-quality', methods=['POST'])
def check_photo_quality():
    try:
        image = request.files.get('image')
        image_comp = request.files.get('image_comp')

        if not image or not image_comp:
            return jsonify({"error": "Both 'image' and 'image_comp' files are required"}), 400

        # Read the image data as bytes
        image_data = image.read()
        image_comp_data = image_comp.read()

        # Validate images
        if not validate_image(image_data) or not validate_image(image_comp_data):
            return jsonify({"error": "One or both images are invalid"}), 400

        # Evaluate photo quality
        result = photo_quality(image_data, image_comp_data)

        return jsonify({"message": result}), 200

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

def validate_image(image_data):
    try:
        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return False
        return True
    except Exception:
        return False

if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0', debug=True)
