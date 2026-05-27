import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights
import serial
import time

# =========================
# SETTINGS
# =========================
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# CONNECT TO ESP32
# =========================
try:
    esp32 = serial.Serial('COM3', 115200)
    time.sleep(2)
    esp32_connected = True
except:
    esp32_connected = False

# =========================
# CLASS NAMES
# =========================
class_names = [
    "biodegradable",
    "recyclable",
    "residual"
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
st.title("♻️ Waste Classification AI")

# ESP32 STATUS
if esp32_connected:
    st.success("ESP32 Connected")
else:
    st.error("ESP32 Not Connected")

option = st.radio(
    "Choose Image Source",
    ["Upload Image", "Use Camera"]
)

uploaded_file = None

if option == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload Waste Image",
        type=["jpg", "jpeg", "png"]
    )

elif option == "Use Camera":

    uploaded_file = st.camera_input(
        "Take a Picture"
    )

# =========================
# PREDICTION
# =========================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(img_tensor)

        predicted = torch.argmax(outputs, dim=1).item()

    result = class_names[predicted]

    # DISPLAY RESULT
    st.success(f"Prediction: {result}")

    # =========================
    # SEND TO ESP32
    # =========================
    if esp32_connected:

        esp32.write((result + "\n").encode())

        st.info("Prediction sent to ESP32")

    # =========================
    # LIVE NOTIFICATIONS
    # =========================
    if result == "biodegradable":

        st.success("🌱 Biodegradable Waste Detected")

    elif result == "residual":

        st.warning("⚠️ Residual Waste Detected")

    elif result == "recyclable":

        st.info("♻️ Recyclable Waste Detected")
