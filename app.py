from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

app = Flask(__name__)

# Set up Inference client
custom_configuration = InferenceConfiguration(confidence_threshold=0.1)
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="vlb86oyC1Fuerrheol9q"
)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # Perform inference directly with the file stream
            with CLIENT.use_configuration(custom_configuration):
                result = CLIENT.infer(file.stream, model_id="billing-8eaq6/5")
            
            # Extract class name from the result
            if result['predictions']:
                class_name = result['predictions'][0]['class']
            else:
                class_name = 'No predictions available'
            
            print(class_name)

            return jsonify({'prediction': class_name}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
