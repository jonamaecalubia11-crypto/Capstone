from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights
import io

# =========================
# SETTINGS
# =========================
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# CLASS NAMES
# =========================
class_names = [
    "biodegradable",
    "non_biodegradable",
    "recyclable"
]

# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# LOAD MODEL
# =========================
model = models.mobilenet_v2(
    weights=MobileNet_V2_Weights.DEFAULT
)

model.classifier[1] = nn.Linear(
    model.last_channel,
    len(class_names)
)

model.load_state_dict(
    torch.load(
        "waste_classifier_best.pth",
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()

# =========================
# FLASK APP
# =========================
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Waste Classification API',
        'endpoints': {
            'POST /predict': 'Upload an image to classify waste. Send image in form field named "image".'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():

    if 'image' not in request.files:
        return jsonify({
            'error': 'No image uploaded'
        })

    file = request.files['image']

    image = Image.open(file.stream).convert("RGB")

    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image)

        predicted = torch.argmax(outputs, dim=1).item()

    result = class_names[predicted]

    return jsonify({
        'prediction': result
    })

# =========================
# RUN SERVER
# =========================
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000
    )