import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights
import time

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="ECOBIN AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>

/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main App */
.stApp {
    background: linear-gradient(
        135deg,
        #eefbf3,
        #d7f5e3,
        #c8f0d7
    );
}

/* Padding */
.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Navbar */
.navbar {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(10px);
    padding: 20px 30px;
    border-radius: 24px;
    box-shadow: 0px 6px 25px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

/* Logo */
.logo-container {
    display: flex;
    align-items: center;
    gap: 18px;
}

.logo-icon {
    font-size: 55px;
    line-height: 1;
}

.logo-text {
    font-size: 38px;
    font-weight: 800;
    color: #14532d;
    line-height: 1.1;
}

.logo-sub {
    color: #4b5563;
    font-size: 15px;
    margin-top: 4px;
}


/* Cards */
.card {
    background: rgba(255,255,255,0.90);
    backdrop-filter: blur(8px);
    padding: 28px;
    border-radius: 28px;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.08);
    height: 100%;
}

/* Section Title */
.section-title {
    font-size: 24px;
    font-weight: 700;
    color: #14532d;
    margin-bottom: 18px;
}

/* Radio Buttons */
.stRadio > div {
    flex-direction: row;
    gap: 20px;
}

/* File Upload */
[data-testid="stFileUploader"] {
    background: #f9fafb;
    border-radius: 18px;
    padding: 12px;
}

/* Buttons */
.stButton>button {
    width: 100%;
    border-radius: 16px;
    border: none;
    height: 3.2em;
    background: linear-gradient(
        to right,
        #166534,
        #22c55e
    );
    color: white;
    font-size: 17px;
    font-weight: bold;
}

/* Image */
img {
    border-radius: 20px;
}

/* Result Card */
.result-card {
    padding: 30px;
    border-radius: 26px;
    color: white;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.15);
}

/* Confidence */
.confidence {
    font-size: 20px;
    margin-top: 10px;
}

/* Tip Box */
.tip-box {
    background: rgba(255,255,255,0.2);
    padding: 12px;
    border-radius: 14px;
    margin-top: 20px;
    font-size: 16px;
}

/* Metrics */
.metric-box {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(6px);
    padding: 22px;
    border-radius: 22px;
    text-align: center;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.06);
}

/* Footer */
.footer {
    text-align: center;
    color: gray;
    margin-top: 30px;
    padding-bottom: 15px;
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
# NAVBAR
# =====================================================
st.markdown("""
<div class="logo-container">

    <div class="logo-icon">
        ♻️
    </div>

    <div>

        <div class="logo-text">
            ECOBIN AI
        </div>

        <div class="logo-sub">
            Smart AI Powered Waste Classification System
        </div>

    </div>

</div>
""", unsafe_allow_html=True)
# =====================================================
# MAIN LAYOUT
# =====================================================
left_col, right_col = st.columns([1.1, 1])

# =====================================================
# LEFT PANEL
# =====================================================
with left_col:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">📤 Upload Waste Image</div>',
        unsafe_allow_html=True
    )

    option = st.radio(
        "Choose Input Method",
        ["📁 Upload Image", "📷 Use Camera"]
    )

    uploaded_file = None

    if option == "📁 Upload Image":

        uploaded_file = st.file_uploader(
            "Upload a waste image",
            type=["jpg", "jpeg", "png"]
        )

    else:

        uploaded_file = st.camera_input(
            "Capture Waste Image"
        )

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# RIGHT PANEL
# =====================================================
with right_col:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">🤖 AI Classification Result</div>',
        unsafe_allow_html=True
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            use_container_width=True
        )

        with st.spinner("ECOBIN AI is analyzing the image..."):

            time.sleep(1.2)

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
        # RESULT DESIGN
        # =====================================================
        if result == "biodegradable":

            color = "#15803d"
            emoji = "🌿"
            tip = "This waste can naturally decompose and may be composted."

        elif result == "recyclable":

            color = "#1d4ed8"
            emoji = "♻️"
            tip = "This item can be processed and reused through recycling."

        else:

            color = "#b91c1c"
            emoji = "🗑️"
            tip = "Dispose this item properly in residual waste bins."

        # =====================================================
        # RESULT CARD
        # =====================================================
        st.markdown(
            f"""
            <div class="result-card"
                 style="background:{color};">

                <h1 style="font-size:70px;">
                    {emoji}
                </h1>

                <h2 style="
                    margin-bottom:5px;
                    font-size:32px;
                ">
                    {result.upper()}
                </h2>

                <div class="confidence">
                    Confidence Score:
                    <b>{confidence_score:.2f}%</b>
                </div>

                <div class="tip-box">
                    💡 {tip}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        # =====================================================
        # PREDICTION SCORES
        # =====================================================
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            '<div class="section-title">📊 Prediction Scores</div>',
            unsafe_allow_html=True
        )

        for i, label in enumerate(class_names):

            score = probs[0][i].item() * 100

            st.write(f"### {label.title()} — {score:.2f}%")

            st.progress(int(score))

    else:

        st.markdown("""
        <div style="
            text-align:center;
            padding:70px 20px;
            color:#6b7280;
        ">
            <h2>📸 No Image Uploaded</h2>

            <p style="font-size:17px;">
                Upload or capture a waste image to start
                AI classification using ECOBIN.
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
        <h2>⚡ Real-Time AI</h2>
        <p>Fast and intelligent waste detection</p>
    </div>
    """, unsafe_allow_html=True)

with m2:

    st.markdown("""
    <div class="metric-box">
        <h2>🧠 MobileNetV2</h2>
        <p>Powered by Deep Learning Technology</p>
    </div>
    """, unsafe_allow_html=True)

with m3:

    st.markdown("""
    <div class="metric-box">
        <h2>🌎 Smart Segregation</h2>
        <p>Helping create a cleaner environment</p>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="footer">
    ECOBIN AI © 2026 | Smart Waste Management System
</div>
""", unsafe_allow_html=True)
