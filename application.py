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

# ===============================
# PAGE CONFIGURATION
# ===============================
st.set_page_config(
    page_title="FreshScan",
    page_icon="🍎",
    layout="centered"
)

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
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
    
    # Reframing models as "Scanning Modes" for the user
    MODEL_CONFIG = {
        "Standard Mode (Faster)": {
            "arch": "mobilenet",
            "file": "mobilenet_freshness_v2_weighted.pth"
        },
        "Deep Scan (High Accuracy)": {
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
        "Menu",
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
# LOAD MODEL (CACHE ENABLED)
# ===============================
@st.cache_resource
def load_model(mode_name):
    config = MODEL_CONFIG[mode_name]
    try:
        if config["arch"] == "mobilenet":
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            model.classifier[1] = nn.Sequential(
                nn.Linear(model.last_channel, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 2)
            )
        elif config["arch"] == "efficientnet":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            model.classifier[1] = nn.Linear(1280, 2)

        state_dict = torch.load(config["file"], map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Scanner error: {e}")
        st.stop()

model = load_model(selected_mode)

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
    1. **Snap or Upload:** Take a photo of your produce.
    2. **AI Analysis:** Our trained neural network scans for imperfections.
    3. **Result:** Get an instant safety recommendation.
    """)
    
    if st.button("Start Scanning Now →"):
        st.info("Please select 'Start Scanning' from the sidebar menu.")

# ===============================
# SCANNER PAGE
# ===============================
elif page == "Start Scanning":
    st.title("📸 Produce Scanner")
    st.write("Point your camera at a single fruit or vegetable or upload a clear photo.")

    # User-centric interaction: Tabs for Camera vs Upload
    input_tab1, input_tab2 = st.tabs(["🤳 Use Camera", "📁 Upload File"])
    
    with input_tab1:
        cam_file = st.camera_input("Scan your food")
    with input_tab2:
        up_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    final_file = cam_file if cam_file else up_file

    if final_file:
        image = Image.open(final_file).convert("RGB")
        st.image(image, caption="Current Preview", use_container_width=True)

        if show_details:
            st.caption(f"Specs: {image.size[0]}x{image.size[1]} pixels")

        # Visual scanning feedback
        with st.status("🔍 Analyzing food quality...", expanded=True) as status:
            time.sleep(1.2) # Artificial delay for "feel"
            img_tensor = image_transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probs, 1)
            
            label = CLASS_NAMES[predicted_class.item()]
            score = confidence.item()
            status.update(label="Scanning Complete!", state="complete", expanded=False)

        # Result Cards
        st.divider()
        if label == "Fresh":
            st.success(f"### ✨ Result: Fresh")
            st.write(f"The scanner is **{score*100:.1f}%** confident that this item is safe to eat.")
            st.balloons()
        else:
            st.error(f"### ⚠️ Result: Spoilage Detected")
            st.write(f"The scanner is **{score*100:.1f}%** confident that this item is showing signs of decay.")
            st.warning("**Recommendation:** We suggest you do not consume this item. Consider composting if possible.")

        if show_confidence:
            st.progress(score)
            
        if score < 0.75:
            st.info("💡 **Tip:** The lighting seems a bit off. For better results, try a brighter environment.")

# ===============================
# ABOUT PAGE
# ===============================
elif page == "About":
    st.title("ℹ️ About FreshScan")
    
    st.markdown("""
    ### Our Mission
    FreshScan was created to bridge the gap between AI technology and daily kitchen safety. 
    By identifying decay early, we help families reduce waste and eat healthier.
    
    ### The Science
    Built using **Convolutional Neural Networks (CNNs)**, the app recognizes visual 
    patterns associated with mold, oxidation, and bruising.
    
    ### The Team
    - **Galoh Safiya Binti Hasnul Hadi**
    - **Marsha Binti Lana Azman**
    - **Nur Arissa Hanani Binti Mohamed Adzlan**
    """)
    st.success("App Status: System Healthy")