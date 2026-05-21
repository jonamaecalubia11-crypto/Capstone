import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Waste Classification AI",
    page_icon="♻️",
    layout="centered"
)

# =========================
# CUSTOM UI DESIGN
# =========================
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #2E8B57;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
    margin-bottom: 30px;
}

.stButton>button {
    width: 100%;
    border-radius: 12px;
    height: 3em;
    background-color: #2E8B57;
    color: white;
    font-size: 18px;
    border: none;
}

.stRadio > div {
    background-color: white;
    padding: 10px;
    border-radius: 10px;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: white;
}

.footer {
    text-align: center;
    margin-top: 40px;
    color: gray;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

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
    "residual",
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
@st.cache_resource
def load_model():

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

    return model

model = load_model()

# =========================
# HEADER
# =========================
st.markdown(
    '<div class="title">♻️ Waste Classification AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload or capture a waste image and let AI classify it instantly.</div>',
    unsafe_allow_html=True
)

# =========================
# IMAGE SOURCE
# =========================
option = st.radio(
    "Choose Image Source",
    ["📁 Upload Image", "📷 Use Camera"]
)

uploaded_file = None

if option == "📁 Upload Image":

    uploaded_file = st.file_uploader(
        "Upload Waste Image",
        type=["jpg", "jpeg", "png"]
    )

elif option == "📷 Use Camera":

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

    # Loading animation
    with st.spinner("Analyzing waste image..."):
        time.sleep(1)

        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            outputs = model(img_tensor)

            probabilities = torch.softmax(outputs, dim=1)

            confidence, predicted = torch.max(
                probabilities,
                dim=1
            )

    result = class_names[predicted.item()]
    confidence_score = confidence.item() * 100

    # =========================
    # RESULT COLORS
    # =========================
    if result == "biodegradable":
        color = "#28a745"
        emoji = "🌿"

    elif result == "recyclable":
        color = "#007bff"
        emoji = "♻️"

    else:
        color = "#dc3545"
        emoji = "🗑️"

    # =========================
    # RESULT DISPLAY
    # =========================
    st.markdown(
        f"""
        <div class="result-box" style="background-color:{color};">
            {emoji}<br>
            {result.upper()}
            <br><br>
            Confidence: {confidence_score:.2f}%
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# FOOTER
# =========================
st.markdown(
    '<div class="footer">Developed using Streamlit + PyTorch + MobileNetV2</div>',
    unsafe_allow_html=True
)
