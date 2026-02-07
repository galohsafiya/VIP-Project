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
.result-card {
    background-color: white;
    padding: 1.2rem;
    border-radius: 10px;
    border-left: 6px solid #4CAF50;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    margin-top: 1rem;
}
.result-card.bad {
    border-left-color: #e74c3c;
}
.result-title {
    font-size: 1.2rem;
    font-weight: 600;
}
.result-sub {
    color: #555;
    margin-top: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# SIDEBAR: APP SETTINGS
# ===============================
with st.sidebar:
    st.title("🛠️ Scanner Settings")

    # NEW: User type selection
    user_type = st.radio(
        "You are a:",
        ["Individual Consumer", "Food Supplier"]
    )

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
    st.title("🍎 Welcome to FreshScan")
    st.subheader("Smart food checking made simple and reliable.")

    st.markdown("""
    FreshScan is an **AI-powered assistant** designed to help users quickly assess the
    **freshness of fruits and vegetables** using image analysis.

    Whether you are an **individual consumer** or part of a **food supply business**,
    FreshScan provides supportive insights to help you make better decisions and
    reduce food waste.
    """)

    st.divider()

    st.markdown("""
    ### 🧠 What FreshScan Does
    - 📷 Analyzes images of fruits and vegetables  
    - 🥗 Classifies freshness as **Fresh** or **Rotten**  
    - 🏷️ Identifies the type of fruit or vegetable  
    - 👤 Adapts results based on **user role** (Consumer or Supplier)  
    """)

    st.divider()

    st.markdown("""
    ### 👥 Who Is This For?
    - **Individual Consumers**  
      Check food safety before consumption at home  

    - **Food Suppliers**  
      Perform quick quality checks before sale or storage  
    """)

    st.divider()

    st.markdown("""
    ### 🔍 How It Works
    1. Select your **user role** and preferred scanning mode from the sidebar  
    2. Upload or capture an image in **Start Scanning**  
    3. The system analyzes the image using deep learning models  
    4. Freshness status, food type, and confidence level are displayed  
    """)

    st.markdown("### 🚀 Get Started")
    st.write(
        "Choose your role from the sidebar, then head to **Start Scanning** "
        "to analyze your produce."
    )

    st.divider()

    st.caption(
        "ℹ️ FreshScan provides AI-assisted visual analysis. "
        "Results may be affected by image quality, lighting conditions, "
        "or visual similarities between food items. "
        "Users are advised to always perform a manual inspection."
    )

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
            st.caption(f"Specs: {image.size[0]} × {image.size[1]} pixels")

        # -------------------------------
        # ANALYSIS
        # -------------------------------
        with st.status("🔍 Analyzing food quality...", expanded=True) as status:
            time.sleep(1)

            img_tensor = image_transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = freshness_model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, pred_class = torch.max(probs, 1)

            label = CLASS_NAMES[pred_class.item()]
            score = confidence.item()
            food_name = predict_food_type(image).title()

            status.update(
                label="Scanning Complete!",
                state="complete",
                expanded=False
            )

        st.divider()

        # -------------------------------
        # RESULT CARD
        # -------------------------------
        if user_type == "Individual Consumer":
            if label == "Fresh":
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-title">✨ Fresh</div>
                    <div class="result-sub">{food_name} appears safe for consumption.</div>
                </div>
                """, unsafe_allow_html=True)

                st.success("Recommendation: Safe to eat based on visual inspection.")
                st.balloons()

            else:
                st.markdown(f"""
                <div class="result-card bad">
                    <div class="result-title">⚠️ Spoilage Detected</div>
                    <div class="result-sub">{food_name} shows visible signs of decay.</div>
                </div>
                """, unsafe_allow_html=True)

                st.warning("Recommendation: Do not consume this item.")

        else:  # Food Supplier
            if label == "Fresh":
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-title">✅ Quality Check: Passed</div>
                    <div class="result-sub">{food_name} meets freshness standards.</div>
                </div>
                """, unsafe_allow_html=True)

                st.info("Action: Suitable for sale or short-term storage.")

            else:
                st.markdown(f"""
                <div class="result-card bad">
                    <div class="result-title">❌ Quality Check: Failed</div>
                    <div class="result-sub">{food_name} does not meet quality standards.</div>
                </div>
                """, unsafe_allow_html=True)

                st.warning("Action: Remove from inventory immediately.")

        # -------------------------------
        # CONFIDENCE 
        # -------------------------------
        if show_confidence:
            st.progress(score)

            if score >= 0.85:
                st.caption("🟢 High confidence result")
            elif score >= 0.65:
                st.caption("🟡 Medium confidence — verify manually")
            else:
                st.caption("🔴 Low confidence — manual inspection required")

        # -------------------------------
        # DISCLAIMER
        # -------------------------------
        st.info(
            "ℹ️ AI-based analysis may be affected by lighting conditions, image quality, "
            "or visual similarities between food items.\n\n"
            "If multiple fruits or vegetables appear in a single image, the system provides "
            "a general assessment and may not correctly detect spoiled areas on individual items "
            "separately.\n\n"
            "Always perform a manual inspection before consumption, sale, or distribution!"
        )

# ===============================
# ABOUT PAGE
# ===============================
elif page == "About":
    st.title("ℹ️ About FreshScan")
    st.subheader("An AI-assisted approach to food freshness assessment.")

    st.markdown("""
    FreshScan was developed as part of an academic project to explore how
    **computer vision and deep learning** can be applied to real-world
    food safety and waste reduction challenges.

    The application provides **visual-based freshness assessment**
    to support decision-making for both **individual consumers**
    and **food suppliers**.
    """)

    st.divider()

    st.markdown("""
    ### 🧠 System Overview
    FreshScan integrates multiple AI components:
    - **Freshness Classification Model**  
      A convolutional neural network (CNN) trained to classify food items as
      *Fresh* or *Rotten* based on visual features.

    - **Food Type Detection Model**  
      A pre-trained image classification model that identifies the type of
      fruit or vegetable present in the image.

    - **Role-Based Output Design**  
      Results and recommendations are adapted depending on whether the user
      is an individual consumer or a food supplier.
    """)

    st.divider()

    st.markdown("""
    ### ⚠️ System Limitations
    - The system performs **image-level classification**, not object detection.  
      When multiple food items appear in a single image, freshness is assessed
      **globally**, not per individual item.

    - Predictions may be affected by:
      - Lighting conditions  
      - Image quality  
      - Visual similarities between different food items  

    FreshScan is intended as a **decision-support tool** and should not replace
    proper manual inspection.
    """)

    st.divider()

    st.markdown("""
    ### 👨‍👩‍👧‍👦 Project Team
    - **Galoh Safiya Binti Hasnul Hadi**  
    - **Marsha Binti Lana Azman**  
    - **Nur Arissa Hanani Binti Mohamed Adzlan**
    """)

    st.markdown("""
    ### 🎓 Academic Context
    This application was developed for the **Visual Information Processing**
    course as part of a group-based project, focusing on:
    - Practical AI deployment
    - Responsible system design
    - User-centered interface development
    """)

    st.success("System Status: Running!")