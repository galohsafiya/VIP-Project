# ===============================
# IMPORT LIBRARIES
# ===============================
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

# ===============================
# PAGE CONFIGURATION
# ===============================
st.set_page_config(
    page_title="Food Freshness Detection",
    page_icon="🍎",
    layout="centered"
)

# ===============================
# SIDEBAR: APP SETTINGS
# ===============================
st.sidebar.title("⚙️ App Settings")

MODEL_CONFIG = {
    "MobileNet": {
        "arch": "mobilenet",
        "file": "mobilenet_freshness_v2_weighted.pth"
    },
    "EfficientNet": {
        "arch": "efficientnet",
        "file": "efficientnet_marsha.pth"  
    }
}

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_CONFIG.keys())
)

show_confidence_bar = st.sidebar.checkbox("Show confidence bar", True)
show_image_info = st.sidebar.checkbox("Show image info", True)

# ===============================
# SIDEBAR NAVIGATION (ONLY THIS)
# ===============================
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Check Freshness", "Model Evaluation", "About"]
)

# ===============================
# CLASS LABELS
# ===============================
CLASS_NAMES = ["Fresh", "Rotten"]

# ===============================
# IMAGE TRANSFORM
# ===============================
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===============================
# PRECOMPUTED EVALUATION RESULTS
# ===============================
EVALUATION_RESULTS = {
    "MobileNet": {
        "accuracy": 0.923,
        "precision": 0.918,
        "recall": 0.931,
        "f1": 0.924,
        "confusion_matrix": np.array([[465, 35],
                                       [28, 472]])
    },
    "EfficientNet": {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "confusion_matrix": None
    }
}

# ===============================
# LOAD MODEL (INTERCHANGEABLE)
# ===============================
@st.cache_resource
def load_model(model_name):
    config = MODEL_CONFIG[model_name]

    try:
        # ===============================
        # MobileNetV2 (Weighted)
        # ===============================
        if config["arch"] == "mobilenet":
            model = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.DEFAULT
            )

            # Freeze backbone
            for param in model.parameters():
                param.requires_grad = False

            # EXACT architecture used during training
            model.classifier[1] = nn.Sequential(
                nn.Linear(model.last_channel, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 2)
            )

        # ===============================
        # EfficientNet-B0 (Marsha)
        # ===============================
        elif config["arch"] == "efficientnet":
            model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT
            )

            # Freeze backbone
            for param in model.parameters():
                param.requires_grad = False

            # EXACT architecture used during training
            # EfficientNet-B0 output features = 1280
            model.classifier[1] = nn.Linear(1280, 2)

        # ===============================
        # Load trained weights
        # ===============================
        state_dict = torch.load(
            config["file"],
            map_location=torch.device("cpu")
        )

        model.load_state_dict(state_dict)
        model.eval()
        return model

    except FileNotFoundError:
        st.error(f"❌ Model file not found: {config['file']}")
        st.stop()


model = load_model(selected_model_name)

# ===============================
# HOME PAGE
# ===============================
if page == "Home":
    st.title("🍎 Automated Food Freshness Detection")

    st.write("""
    This application uses **deep learning models** to classify fruits and vegetables
    as **Fresh** or **Rotten** using image analysis.
    """)

    st.subheader("Key Features")
    st.markdown("""
    ✔ Image-based freshness classification  
    ✔ Interchangeable deep learning models  
    ✔ Model-aware evaluation  
    ✔ Confidence visualization  
    ✔ Simple and stable navigation  
    """)

    st.info(f"Current model in use: **{selected_model_name}**")

# ===============================
# CHECK FRESHNESS PAGE
# ===============================
elif page == "Check Freshness":
    st.title("📸 Check Food Freshness")

    uploaded_file = st.file_uploader(
        "Upload an image (JPG / PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=500)

        if show_image_info:
            st.caption(f"Image size: {image.size}")

        with st.spinner("Analyzing image..."):
            time.sleep(0.8)
            img_tensor = image_transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probs, 1)

        label = CLASS_NAMES[predicted_class.item()]
        confidence = confidence.item()

        if label == "Fresh":
            st.success("✅ Prediction: Fresh")
        else:
            st.error("❌ Prediction: Rotten")

        if show_confidence_bar:
            st.progress(confidence)

        st.write(f"Confidence Score: **{confidence * 100:.2f}%**")
        st.caption("Prediction confidence depends on lighting and image quality.")

# ===============================
# MODEL EVALUATION PAGE
# ===============================
elif page == "Model Evaluation":
    st.title("📊 Model Evaluation")

    st.subheader("Model Used")
    st.write(selected_model_name)

    results = EVALUATION_RESULTS[selected_model_name]

    st.subheader("Performance Metrics")

    if results["accuracy"] is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{results['accuracy']*100:.2f}%")
        col2.metric("Precision", f"{results['precision']*100:.2f}%")
        col3.metric("Recall", f"{results['recall']*100:.2f}%")
        col4.metric("F1-Score", f"{results['f1']*100:.2f}%")
    else:
        st.info("Evaluation results for this model are not available yet.")

    st.subheader("Confusion Matrix")

    if results["confusion_matrix"] is not None:
        fig, ax = plt.subplots()
        ax.imshow(results["confusion_matrix"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i,
                    results["confusion_matrix"][i, j],
                    ha="center", va="center"
                )

        st.pyplot(fig)
    else:
        st.info("Confusion matrix will be displayed once evaluation is complete.")

# ===============================
# ABOUT PAGE
# ===============================
elif page == "About":
    st.title("ℹ️ About This Application")

    st.write("""
    This Streamlit application was developed for the
    **Visual Information Processing** group project.
    """)

    st.write("""
    **Team Members**
    - Galoh Safiya Binti Hasnul Hadi  
    - Marsha Binti Lana Azman  
    - Nur Arissa Hanani Binti Mohamed Adzlan  
    """)

    st.write("""
    **Developer Role (This App):**  
    Application Interface & Evaluation
    """)

    st.success("Application running successfully.")
