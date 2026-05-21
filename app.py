import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights

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
# STREAMLIT UI
# =========================
st.title("Waste Classification AI")

uploaded_file = st.file_uploader(
    "Upload Waste Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(img_tensor)

        predicted = torch.argmax(outputs, dim=1).item()

    result = class_names[predicted]

    st.success(f"Prediction: {result}")
