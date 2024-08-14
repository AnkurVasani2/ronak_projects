from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
# Set up Inference client
custom_configuration = InferenceConfiguration(confidence_threshold=0.1)
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="vlb86oyC1Fuerrheol9q"
)

@app.route('/upload', methods=["POST"])
def upload():
    if request.method == "POST":
        imagefile = request.files['file']
        filename = "file.jpeg"  # Save the file with the same name for consistency

        # Save the uploaded image
        imagefile.save(filename)

        try:
            # Infer using the InferenceHTTPClient
            with CLIENT.use_configuration(custom_configuration):
                result = CLIENT.infer(filename, model_id="billing-8eaq6/5")
            if result['predictions']:
                class_name = result['predictions'][0]['class']
            else:
                class_name = 'No predictions available'
            
            print(class_name)

            return jsonify({'prediction': class_name}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
