import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import zipfile
import os
import numpy as np
import matplotlib.pyplot as plt
import json

# =====================
# Load Model + Config
# =====================
MODEL_PATH = "deploy_resnet50_slide.pth"
CONFIG_PATH = "deploy_resnet50_slide_threshold.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Load threshold config
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

THRESHOLD = config.get("threshold", 0.5)
PERCENTILE = config.get("percentile", 90)

# =====================
# Transforms
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =====================
# Streamlit UI
# =====================
st.title("🧬 Histopathology Classifier")
st.write(f"Model loaded with config: percentile @ {PERCENTILE} | Threshold = {THRESHOLD:.2f}")

uploaded_file = st.file_uploader("Upload a histopathology image or zip of patches", type=["jpg", "png", "jpeg", "zip"])

patch_probs = []  # Initialize globally

if uploaded_file is not None:
    if uploaded_file.name.endswith(".zip"):
        # Handle multiple patch images
        with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
            zip_ref.extractall("temp_images")

        image_files = [os.path.join("temp_images", f) for f in os.listdir("temp_images") if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        st.write("Running inference on all patches... Please wait ⏳")

        for img_path in image_files:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                outputs = model(tensor)
                prob = torch.softmax(outputs, dim=1)[:, 1].item()
                patch_probs.append(prob)

        if patch_probs:
            agg_prob = np.mean(patch_probs)
            label = "Malignant" if agg_prob >= THRESHOLD else "Benign"
            emoji = "🔴" if label == "Malignant" else "🟢"

            st.subheader("📊 Slide-level Result")
            st.write(f"{emoji} {label} (Aggregated Probability = {agg_prob:.2f}, Threshold = {THRESHOLD:.2f})")

            st.subheader("📈 Patch-level Summary")
            st.write(f"Number of patches processed: {len(patch_probs)}")
            st.write(f"Patch probabilities → min: {np.min(patch_probs):.2f}, mean: {np.mean(patch_probs):.2f}, max: {np.max(patch_probs):.2f}")

            # Plot histogram
            fig, ax = plt.subplots()
            ax.hist(patch_probs, bins=10, color="skyblue", edgecolor="black")
            ax.set_xlabel("Probability of Malignant")
            ax.set_ylabel("Number of patches")
            st.pyplot(fig)
        else:
            st.warning("⚠️ No patches were processed from the zip file.")

    else:
        # Handle single image
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(tensor)
            prob = torch.softmax(outputs, dim=1)[:, 1].item()

        label = "Malignant" if prob >= THRESHOLD else "Benign"
        emoji = "🔴" if label == "Malignant" else "🟢"

        st.subheader("📊 Prediction Result")
        st.write(f"{emoji} {label} (Probability = {prob:.2f}, Threshold = {THRESHOLD:.2f})")