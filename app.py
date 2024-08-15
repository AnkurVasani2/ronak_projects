from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
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


def compress_image(input_path, output_path, max_size=(1024, 1024), quality=85):
    with Image.open(input_path) as img:
        img.thumbnail(max_size)
        img.save(output_path, "JPEG", quality=quality)
@app.route('/upload', methods=["POST"])
def upload():
    if request.method == "POST":
        imagefile = request.files['file']
        filename = "file.jpeg"  # Save the file with the same name for consistency

        # Save the uploaded image
        imagefile.save(filename)
        compress_image(filename, filename)
        try:
            # Infer using the InferenceHTTPClient
            with CLIENT.use_configuration(custom_configuration):
                result = CLIENT.infer(filename, model_id="billing-8eaq6/6")
            if result['predictions']:
                class_name = result['predictions'][0]['class']
            else:
                class_name = 'No predictions available'
            
            print(class_name)

            return jsonify({'prediction': class_name}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
