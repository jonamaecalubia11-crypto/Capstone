import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights
import numpy as np
import time

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# CUSTOM CSS (REAL APP UI)
# =====================================================
st.markdown("""
<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
}

/* MAIN BACKGROUND */
.stApp {
    background: linear-gradient(to bottom right, #edf7f1, #dff5e5);
}

/* TOP NAVBAR */
.navbar {
    background-color: white;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.logo-text {
    font-size: 34px;
    font-weight: bold;
    color: #1b5e20;
}

.logo-sub {
    color: gray;
    margin-top: -10px;
}

/* CARD */
.card {
    background: white;
    padding: 25px;
    border-radius: 24px;
    box-shadow: 0px 6px 25px rgba(0,0,0,0.08);
}

/* RESULT CARD */
.result-card {
    padding: 30px;
    border-radius: 24px;
    color: white;
    text-align: center;
    margin-top: 20px;
}

/* CONFIDENCE */
.confidence-text {
    font-size: 18px;
    margin-top: 10px;
}

/* SECTION TITLE */
.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 15px;
    color: #1b5e20;
}

/* METRICS */
.metric-box {
    background: #f7f7f7;
    padding: 15px;
    border-radius: 18px;
    text-align: center;
}

/* BUTTON */
.stButton>button {
    width: 100%;
    border-radius: 14px;
    height: 3.2em;
    border: none;
    background: linear-gradient(to right, #1b5e20, #43a047);
    color: white;
    font-size: 17px;
    font-weight: bold;
}

/* RADIO */
.stRadio > div {
    flex-direction: row;
    gap: 20px;
}

/* IMAGE */
img {
    border-radius: 18px;
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
# NAVBAR
# =====================================================
st.markdown("""
<div class="navbar">
    <div class="logo-text">♻️ EcoSort AI</div>
    <div class="logo-sub">
        Smart Waste Classification System
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# LAYOUT
# =====================================================
left_col, right_col = st.columns([1.2, 1])

# =====================================================
# LEFT SIDE
# =====================================================
with left_col:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Upload Waste Image</div>',
        unsafe_allow_html=True
    )

    option = st.radio(
        "Select Source",
        ["📁 Upload", "📷 Camera"]
    )

    uploaded_file = None

    if option == "📁 Upload":

        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"]
        )

    else:

        uploaded_file = st.camera_input(
            "Take a picture"
        )

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# RIGHT SIDE
# =====================================================
with right_col:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">AI Detection Result</div>',
        unsafe_allow_html=True
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            use_container_width=True
        )

        with st.spinner("AI is analyzing the waste..."):

            time.sleep(1.5)

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

        # =====================================================
        # RESULT STYLE
        # =====================================================
        if result == "biodegradable":

            color = "#2e7d32"
            emoji = "🌿"
            tip = "Can be composted naturally."

        elif result == "recyclable":

            color = "#1565c0"
            emoji = "♻️"
            tip = "Send this item for recycling."

        else:

            color = "#c62828"
            emoji = "🗑️"
            tip = "Dispose properly in residual bins."

        # =====================================================
        # RESULT CARD
        # =====================================================
        st.markdown(
            f"""
            <div class="result-card"
                 style="background:{color};">

                <h1>{emoji}</h1>

                <h2 style="margin-bottom:5px;">
                    {result.upper()}
                </h2>

                <div class="confidence-text">
                    Confidence: {confidence_score:.2f}%
                </div>

                <br>

                <div style="
                    background: rgba(255,255,255,0.2);
                    padding:10px;
                    border-radius:12px;
                    font-size:16px;
                ">
                    💡 {tip}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        # =====================================================
        # PROBABILITY SCORES
        # =====================================================
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            '<div class="section-title">Prediction Scores</div>',
            unsafe_allow_html=True
        )

        for i, label in enumerate(class_names):

            score = probs[0][i].item() * 100

            st.progress(int(score))

            st.write(f"{label.title()} — {score:.2f}%")

    else:

        st.markdown("""
        <div style="
            text-align:center;
            padding:60px 20px;
            color:gray;
        ">
            <h2>📸 No Image Selected</h2>
            <p>
                Upload or capture a waste image to begin
                classification.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# BOTTOM DASHBOARD
# =====================================================
st.markdown("<br>", unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)

with m1:

    st.markdown("""
    <div class="metric-box">
        <h2>⚡ Fast AI</h2>
        <p>Real-time waste detection</p>
    </div>
    """, unsafe_allow_html=True)

with m2:

    st.markdown("""
    <div class="metric-box">
        <h2>🧠 MobileNetV2</h2>
        <p>Deep Learning Powered</p>
    </div>
    """, unsafe_allow_html=True)

with m3:

    st.markdown("""
    <div class="metric-box">
        <h2>🌎 Eco Friendly</h2>
        <p>Supports smart segregation</p>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<div style="
    text-align:center;
    color:gray;
    padding-bottom:10px;
">
    EcoSort AI © 2026 | Smart Waste Management System
</div>
""", unsafe_allow_html=True)
