from flask import Flask, request, jsonify
import cv2
import numpy as np
from io import BytesIO
import json
import requests

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
    # Read the image from the in-memory byte stream
    image_array = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    # Resize image to specified dimensions (10x10 by default)
    image_resized = cv2.resize(image_array, (dim, dim))

    # Convert color space to LAB format and extract L channel
    L, A, B = cv2.split(cv2.cvtColor(image_resized, cv2.COLOR_BGR2LAB))

    # Normalize L channel by dividing all pixel values by the maximum pixel value
    L = L / np.max(L)

    # Return True if mean is greater than thresh, indicating "bright"
    return np.mean(L) > thresh

def similarity_check(image_data_1, image_data_2):
    # Decode the images from in-memory byte streams
    img1 = cv2.imdecode(np.frombuffer(image_data_1, np.uint8), 0)  # Load in grayscale
    img2 = cv2.imdecode(np.frombuffer(image_data_2, np.uint8), 0)  # Load in grayscale

    # Check if images are loaded correctly
    if img1 is None or img2 is None:
        raise ValueError(f"One or both images could not be loaded. Check the inputs.")

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Check if descriptors are found
    if des1 is None or des2 is None:
        raise ValueError("Descriptors could not be found in one or both images. They may lack sufficient features.")

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Calculate similarity score
    similarity_score = len(matches) / min(len(kp1), len(kp2))
    return similarity_score

def photo_quality(image_data, image_data_comp):
    if quality_check(BytesIO(image_data)) < 0.55:
        return "Bad quality"

    if not isbright(image_data):
        return "The picture is too dark"

    if similarity_check(image_data, image_data_comp) < 0.35:
        return "The picture doesn't capture the object correctly"

    return "Great picture"

# Flask app definition
app = Flask(__name__)

@app.route('/photo-quality', methods=['POST'])
def check_photo_quality():
    try:
        # Expecting 'image' and 'image_comp' in the request
        image = request.files.get('image')
        image_comp = request.files.get('image_comp')

        if not image or not image_comp:
            return jsonify({"error": "Both 'image' and 'image_comp' files are required"}), 400

        # Read the image data as bytes
        image_data = image.read()
        image_comp_data = image_comp.read()

        # Evaluate photo quality
        result = photo_quality(image_data, image_comp_data)

        return jsonify({"message": result}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # Dynamically get the port from the environment
    app.run(host='0.0.0.0', port=port, debug=False)  # Use debug=False in production
