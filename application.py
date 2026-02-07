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
import cv2  # Added for Grad-CAM processing

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
# GRAD-CAM HELPER FUNCTION
# ===============================
def generate_gradcam(model, img_tensor, original_image, target_layer_name):
    """Generates a heatmap overlay showing where the AI is focused."""
    feature_maps = []
    gradients = []

    def save_feature_map(module, input, output):
        feature_maps.append(output)
    
    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Access the target layer
    target_layer = dict(model.named_modules())[target_layer_name]
    
    handle_forward = target_layer.register_forward_hook(save_feature_map)
    handle_backward = target_layer.register_full_backward_hook(save_gradient)

    # Forward pass
    model.zero_grad()
    output = model(img_tensor)
    _, pred_class = torch.max(output, 1)
    
    # Backward pass
    output[:, pred_class].backward()

    # Process gradients and feature maps
    grads = gradients[0].cpu().data.numpy()
    f_maps = feature_maps[0].cpu().data.numpy()[0]
    
    handle_forward.remove()
    handle_backward.remove()

    weights = np.mean(grads, axis=(2, 3))[0]
    cam = np.zeros(f_maps.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * f_maps[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    img_bg = np.array(original_image.resize((224, 224)))
    overlayed = cv2.addWeighted(img_bg, 0.6, heatmap, 0.4, 0)
    
    return overlayed

# ===============================
# SIDEBAR: APP SETTINGS
# ===============================
with st.sidebar:
    st.title("🛠️ Scanner Settings")

    user_type = st.radio(
        "You are a:",
        ["Individual Consumer", "Food Supplier"]
    )

    MODEL_CONFIG = {
        "Standard Mode (Faster)": {
            "arch": "mobilenet",
            "file": "mobilenet_freshness_v2_weighted.pth",
            "layer": "features.18" # Final conv layer for MobileNetV2
        },
        "Deep Scan (More Accurate)": {
            "arch": "efficientnet",
            "file": "efficientnet_marsha.pth",
            "layer": "features.8"  # Final conv layer for EfficientNet
        }
    }

    selected_mode = st.selectbox(
        "Scanning Accuracy",
        list(MODEL_CONFIG.keys())
    )

    with st.expander("Advanced Options"):
        show_confidence = st.checkbox("Show confidence rating", True)
        show_details = st.checkbox("Show image metadata", False)
        enable_gradcam = st.checkbox("Highlight detection zones", True)

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
        # Save target layer to model instance
        model.target_layer_name = config["layer"]
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
    processor = AutoImageProcessor.from_pretrained("jazzmacedo/fruits-and-vegetables-detector-36")
    model = AutoModelForImageClassification.from_pretrained("jazzmacedo/fruits-and-vegetables-detector-36")
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
    - 🥗 Classifies freshness as **Fresh** or **Rotten** - 🏷️ Identifies the type of fruit or vegetable  
    - 👤 Adapts results based on **user role** (Consumer or Supplier)  
    """)

    st.divider()

    st.markdown("""
    ### 👥 Who Is This For?
    - **Individual Consumers** Check food safety before consumption at home  

    - **Food Suppliers** Perform quick quality checks before sale or storage  
    """)

    st.divider()

    st.markdown("""
    ### 🔍 How It Works
    1. Select your **user role** and preferred scanning mode from the sidebar  
    2. Upload or capture an image in **Start Scanning** 3. The system analyzes the image using deep learning models  
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
        up_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

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
            img_tensor.requires_grad = True # Required for Grad-CAM

            # Freshness Prediction
            outputs = freshness_model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_class = torch.max(probs, 1)

            label = CLASS_NAMES[pred_class.item()]
            score = confidence.item()
            food_name = predict_food_type(image).title()

            # Generate Grad-CAM if enabled
            if enable_gradcam:
                heatmap_img = generate_gradcam(freshness_model, img_tensor, image, freshness_model.target_layer_name)

            status.update(label="Scanning Complete!", state="complete", expanded=False)

        st.divider()

        # -------------------------------
        # GRAD-CAM VISUALIZATION
        # -------------------------------
        if enable_gradcam:
            st.subheader("🔍 AI Analysis Heatmap")
            st.image(heatmap_img, caption="Red zones indicate where the AI detected freshness or decay markers.", use_container_width=True)
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

        st.info(
            "ℹ️ AI-based analysis may be affected by lighting conditions, image quality, "
            "or visual similarities between food items.\n\n"
            "If multiple fruits or vegetables appear in a single image, the system provides "
            "a general assessment.\n\n"
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
    """)

    st.divider()

    st.markdown("""
    ### 🧠 System Overview
    FreshScan integrates multiple AI components:
    - **Freshness Classification Model** A convolutional neural network (CNN) trained to classify food items as
      *Fresh* or *Rotten*.
      
    - **Explainable AI (Grad-CAM)** The system uses Gradient-weighted Class Activation Mapping to visually 
      highlight the specific pixels that influenced the AI's decision.

    - **Food Type Detection Model** Identifies the type of fruit or vegetable present in the image.
    """)

    st.divider()

    st.markdown("""
    ### 👨‍👩‍👧‍👦 Project Team
    - **Galoh Safiya Binti Hasnul Hadi** - **Marsha Binti Lana Azman** - **Nur Arissa Hanani Binti Mohamed Adzlan**
    """)

    st.success("System Status: Running!")
