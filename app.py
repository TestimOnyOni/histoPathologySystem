import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import os

# ================================
# Page Config (must be first)
# ================================
st.set_page_config(page_title="BreakHis Classifier", page_icon="🧬")

# ================================
# Settings
# ================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "deploy_resnet50_slide.pth"
CONFIG_PATH = "deploy_resnet50_slide_threshold.json"

# ================================
# Model Loader
# ================================
def get_resnet50_model(num_classes=2):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

@st.cache_resource
def load_model_and_config(model_path, config_path):
    # Load model
    model = get_resnet50_model(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    return model, config

model, config = load_model_and_config(MODEL_PATH, CONFIG_PATH)
best_threshold = config.get("best_threshold", 0.5)
agg_method = config.get("agg_method", "percentile")
percentile_value = config.get("percentile", 90)

# ================================
# Transforms
# ================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ================================
# Prediction function
# ================================
def predict_image(image: Image.Image):
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[:, 1].item()  # malignant prob

    # Apply threshold
    label = 1 if probs >= best_threshold else 0
    return label, probs

# ================================
# Streamlit UI
# ================================
st.title("🧬 BreakHis Histopathology Classifier")
st.write(f"Model loaded with config: **{agg_method} @ {percentile_value}** | Threshold = {best_threshold}")

uploaded_file = st.file_uploader("Upload a histopathology image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run prediction
    label, prob = predict_image(image)

    st.subheader("Prediction Result")
    if label == 1:
        st.error(f"⚠️ Malignant (prob={prob:.3f})")
    else:
        st.success(f"✅ Benign (prob={prob:.3f})")