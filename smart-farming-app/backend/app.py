from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import models, transforms
from PIL import Image
import torch
import os

app = Flask(__name__)
CORS(app)  # Allow all origins temporarily for testing

# Load your pre-trained model
model = models.resnet18()
model.load_state_dict(torch.load('resnet18-f37072fd.pth'))
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(image_path):
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

@app.route('/detect-disease', methods=['POST'])
def detect_disease():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        image = request.files['image']
        upload_folder = './uploads/'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        image_path = os.path.join(upload_folder, image.filename)
        image.save(image_path)

        # Predict the disease
        disease = predict(image_path)
        return jsonify({'disease': disease})
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)  # Try a different port 