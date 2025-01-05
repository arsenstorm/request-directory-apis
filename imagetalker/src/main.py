import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify


load_dotenv()

DEBUG_MODE = os.getenv('IMAGETALKER_DEBUG', 'false').lower() == 'true'

RUN_LOCAL = os.getenv('IMAGETALKER_RUN_LOCAL', 'false').lower() == 'true'
# If we're running locally, then we don't need the replicate settings
REPLICATE = {
    "id": os.getenv('IMAGETALKER_REPLICATE_MODEL', ''),
    "key": os.getenv('IMAGETALKER_REPLICATE_API_KEY', ''),
}

app = Flask(__name__)


@app.route('/create', methods=['POST'])
def create():
    if 'image' not in request.files:
        return jsonify({
            "error": "You haven’t included an image in the `image` parameter.",
            "success": False
        }), 400

    image = request.files['image']

    if image.filename == '':
        return jsonify({
            "error": "The image you have uploaded is invalid.",
            "success": False
        }), 400

    if "audio" not in request.files:
        return jsonify({
            "error": "You haven’t included an audio file via the `audio` parameter.",
            "success": False
        }), 400

    audio = request.files["audio"]

    if audio.filename == "":
        return jsonify({
            "error": "The audio file you have uploaded is invalid.",
            "success": False
        }), 400
    
    # If we're running locally, then we use the local model, otherwise we use the replicate model

    try:
        return
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500


if __name__ == '__main__':
    port = int(os.getenv('IMAGETALKER_PORT', 7004))
    app.run(host='0.0.0.0', port=port, debug=DEBUG_MODE)
