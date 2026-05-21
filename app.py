import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="ECOBIN AI",
    layout="wide"
)

# =====================================================
# SIMPLE CLEAN UI
# =====================================================
st.markdown("""
<style>

/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Background */
.stApp {
    background-color: #f5f7f9;
}

/* Main Padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

/* Title */
.main-title {
    font-size: 42px;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 5px;
}

.sub-title {
    color: #6b7280;
    font-size: 16px;
    margin-bottom: 30px;
}

/* Cards */
.card {
    background: white;
    padding: 25px;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* Section Title */
.section-title {
    font-size: 22px;
    font-weight: 600;
    color: #111827;
    margin-bottom: 20px;
}

/* Buttons */
.stButton>button {
    width: 100%;
    height: 3em;
    border-radius: 10px;
    border: none;
    background-color: #166534;
    color: white;
    font-size: 16px;
    font-weight: 600;
}

/* Upload Box */
[data-testid="stFileUploader"] {
    border-radius: 10px;
}

/* Result Box */
.result-box {
    padding: 25px;
    border-radius: 14px;
    text-align: center;
    margin-top: 20px;
    color: white;
}

/* Footer */
.footer {
    text-align: center;
    color: #6b7280;
    margin-top: 40px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SETTINGS
# =====================================================
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# CLASS NAMES
# =====================================================
class_names = [
    "biodegradable",
    "residual",
    "recyclable"
]

# =====================================================
# IMAGE TRANSFORM
# =====================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# LOAD MODEL
# =====================================================
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

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div class="main-title">
    ECOBIN AI
</div>

<div class="sub-title">
    Smart Waste Classification System
</div>
""", unsafe_allow_html=True)

# =====================================================
# LAYOUT
# =====================================================
left_col, right_col = st.columns([1, 1])

# =====================================================
# LEFT PANEL
# =====================================================
with left_col:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Upload Waste Image</div>',
        unsafe_allow_html=True
    )

    option = st.radio(
        "Select Input Method",
        ["Upload Image", "Use Camera"]
    )

    uploaded_file = None

    if option == "Upload Image":

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"]
        )

    else:

        uploaded_file = st.camera_input(
            "Capture Image"
        )

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# RIGHT PANEL
# =====================================================
with right_col:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Classification Result</div>',
        unsafe_allow_html=True
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            use_container_width=True
        )

        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            outputs = model(img_tensor)

            probs = torch.softmax(outputs, dim=1)

            confidence, predicted = torch.max(
                probs,
                dim=1
            )

        result = class_names[predicted.item()]
        confidence_score = confidence.item() * 100

        # RESULT COLORS
        if result == "biodegradable":
            color = "#166534"

        elif result == "recyclable":
            color = "#1d4ed8"

        else:
            color = "#b91c1c"

        # RESULT CARD
        st.markdown(
            f"""
            <div class="result-box"
                 style="background:{color};">

                <h2 style="font-size:30px;">
                    {result.upper()}
                </h2>

                <p style="font-size:18px;">
                    Confidence Score:
                    <b>{confidence_score:.2f}%</b>
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        # PREDICTION SCORES
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            '<div class="section-title">Prediction Scores</div>',
            unsafe_allow_html=True
        )

        for i, label in enumerate(class_names):

            score = probs[0][i].item() * 100

            st.write(f"{label.title()} — {score:.2f}%")

            st.progress(int(score))

    else:

        st.info("Upload or capture an image to start classification.")

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="footer">
    ECOBIN AI © 2026
</div>
""", unsafe_allow_html=True)
