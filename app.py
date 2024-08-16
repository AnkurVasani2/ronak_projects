from flask import Flask, request, jsonify
from io import BytesIO
import base64
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Set up Inference client
custom_configuration = InferenceConfiguration(confidence_threshold=0.1)
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="vlb86oyC1Fuerrheol9q"
)

def compress_image(image_file, max_size=(1024, 1024), quality=85):
    img = Image.open(image_file)
    
    # Convert to RGB if the image has an alpha channel
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        img = img.convert('RGB')
    
    img.thumbnail(max_size)
    
    compressed_img = BytesIO()
    img.save(compressed_img, format="JPEG", quality=quality)
    compressed_img.seek(0)
    
    return base64.b64encode(compressed_img.getvalue()).decode('utf-8')

@app.route('/upload', methods=["POST"])
def upload():
    if request.method == "POST":
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected for uploading'}), 400
        
        if file:
            try:
                # Compress the image and convert to base64
                compressed_img_base64 = compress_image(file)
                
                # Infer using the InferenceHTTPClient
                with CLIENT.use_configuration(custom_configuration):
                    result = CLIENT.infer(compressed_img_base64, model_id="billing-8eaq6/6")
                
                # Collect all class names from the predictions
                class_names = [prediction['class'] for prediction in result.get('predictions', [])]
                
                print(class_names)
                if not class_names:
                    class_names = ['No predictions available']
                
                return jsonify({'predictions': class_names}), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500