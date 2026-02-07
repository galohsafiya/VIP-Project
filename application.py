# ===============================
# IMPORT LIBRARIES
# ===============================
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import time

from transformers import AutoImageProcessor, AutoModelForImageClassification

# ===============================
# PAGE CONFIGURATION
# ===============================
st.set_page_config(
    page_title="FreshScan",
    page_icon="🍎",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    </style>
""", unsafe_allow_html=True)

# ===============================
# SIDEBAR: APP SETTINGS
# ===============================
with st.sidebar:
    st.title("🛠️ Scanner Settings")
    st.info("Adjust the scanner sensitivity and display options below.")

    MODEL_CONFIG = {
        "Standard Mode (Faster)": {
            "arch": "mobilenet",
            "file": "mobilenet_freshness_v2_weighted.pth"
        },
        "Deep Scan (More Accurate)": {
            "arch": "efficientnet",
            "file": "efficientnet_marsha.pth"
        }
    }

    selected_mode = st.selectbox(
        "Scanning Accuracy",
        list(MODEL_CONFIG.keys())
    )

    with st.expander("Advanced Options"):
        show_confidence = st.checkbox("Show confidence rating", True)
        show_details = st.checkbox("Show image metadata", False)

    st.divider()
    page = st.radio(
        "Navigation",
        ["Home", "Start Scanning", "About"]
    )

# ===============================
# CLASS LABELS & TRANSFORMS
# ===============================
CLASS_NAMES = ["Fresh", "Rotten"]

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===============================
# LOAD FRESHNESS MODEL
# ===============================
@st.cache_resource
def load_model(mode_name):
    config = MODEL_CONFIG[mode_name]

    try:
        if config["arch"] == "mobilenet":
            model = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.DEFAULT
            )
            for p in model.parameters():
                p.requires_grad = False

            model.classifier[1] = nn.Sequential(
                nn.Linear(model.last_channel, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 2)
            )

        elif config["arch"] == "efficientnet":
            model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT
            )
            for p in model.parameters():
                p.requires_grad = False

            model.classifier[1] = nn.Linear(1280, 2)

        state_dict = torch.load(
            config["file"],
            map_location=torch.device("cpu")
        )

        model.load_state_dict(state_dict)
        model.eval()
        return model

    except Exception as e:
        st.error(f"Scanner error: {e}")
        st.stop()

freshness_model = load_model(selected_mode)

# ===============================
# LOAD HF FOOD DETECTOR
# ===============================
@st.cache_resource
def load_food_detector():
    processor = AutoImageProcessor.from_pretrained(
        "jazzmacedo/fruits-and-vegetables-detector-36"
    )
    model = AutoModelForImageClassification.from_pretrained(
        "jazzmacedo/fruits-and-vegetables-detector-36"
    )
    model.eval()
    return processor, model


def predict_food_type(image):
    processor, food_model = load_food_detector()

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = food_model(**inputs)
        pred_id = outputs.logits.argmax(-1).item()

    return food_model.config.id2label[pred_id]

# ===============================
# HOME PAGE
# ===============================
if page == "Home":
    st.title("🍎 FreshScan")
    st.subheader("Keep your kitchen safe and reduce food waste.")

    st.markdown("""
    Welcome to **FreshScan**! This tool helps you instantly determine if your
    fruits and vegetables are fresh or showing signs of decay.

    ### How it works:
    1. Click **Start Scanning** on the sidebar navigation.
    2. Upload or capture an image of a fruit or vegetable you would like to identify.
    3. Results receive in seconds! We will analyze its freshness and the food type.
    4. You will also get a recommendation!
    """)

# ===============================
# SCANNER PAGE
# ===============================
elif page == "Start Scanning":
    st.title("📸 Produce Scanner")
    st.write("Upload or capture a clear photo of a single fruit or vegetable.")

    tab1, tab2 = st.tabs(["🤳 Use Camera", "📁 Upload File"])

    with tab1:
        cam_file = st.camera_input("Scan your food")
    with tab2:
        up_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"]
        )

    final_file = cam_file if cam_file else up_file

    if final_file:
        image = Image.open(final_file).convert("RGB")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Current Preview", width=350)

        if show_details:
            st.caption(f"Specs: {image.size[0]} x {image.size[1]} pixels")

        with st.status("🔍 Analyzing food quality...", expanded=True) as status:
            time.sleep(1)

            img_tensor = image_transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = freshness_model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, pred_class = torch.max(probs, 1)

            label = CLASS_NAMES[pred_class.item()]
            score = confidence.item()

            food_name = predict_food_type(image)

            status.update(
                label="Scanning Complete!",
                state="complete",
                expanded=False
            )

        st.divider()

        if label == "Fresh":
            st.success("### ✨ Result: Fresh")
            st.write(
                f"The scanner is **{score*100:.1f}%** confident that this "
                f"**{food_name.lower()}** is safe to eat."
            )
            st.balloons()
        else:
            st.error("### ⚠️ Result: Spoilage Detected")
            st.write(
                f"The scanner is **{score*100:.1f}%** confident that this "
                f"**{food_name.lower()}** is showing signs of decay."
            )
            st.warning(
                "**Recommendation:** We suggest you do not consume this item. "
                "Consider composting if possible."
            )

        if show_confidence:
            st.progress(score)

        if score < 0.75:
            st.info(
                "💡 Tip: Lighting may affect accuracy. "
                "Try a brighter environment."
            )

# ===============================
# ABOUT PAGE
# ===============================
elif page == "About":
    st.title("ℹ️ About FreshScan")

    st.markdown("""
    ### Our Mission
    FreshScan helps reduce food waste and improve food safety
    through AI-powered visual analysis.

    ### Technology
    - CNN-based freshness classification
    - Pretrained food-category detection
    - No additional training required

    ### Team
    - Galoh Safiya Binti Hasnul Hadi
    - Marsha Binti Lana Azman
    - Nur Arissa Hanani Binti Mohamed Adzlan
    """)

    st.success("App Status: System Healthy")