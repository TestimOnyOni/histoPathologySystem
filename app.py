import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import zipfile
import os
import tempfile
import json
import matplotlib.pyplot as plt

# ================================
# Page config
# ================================
st.set_page_config(page_title="BreakHis Slide Classifier", page_icon="🧬", layout="centered")

# ================================
# Page config
# ================================
st.set_page_config(page_title="BreakHis Slide Classifier", page_icon="🧬", layout="centered")

# ================================
# Settings
# ================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_resnet50_balanced.pth"
CONFIG_PATH = "deploy_resnet50_slide_threshold.json"

# Load threshold config
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
THRESHOLD = config.get("best_threshold", 0.39)
PERCENTILE = config.get("percentile", 90)

# ================================
# Model Loader
# ================================
@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

model = load_model()
st.sidebar.success(f"Model ready ✅ | Percentile={PERCENTILE}, Threshold={THRESHOLD}")

# ================================
# Preprocessing
# ================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_patch(img: Image.Image):
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()[0]
    return probs

# ================================
# Slide-level Prediction
# ================================
def predict_slide_from_zip(zip_file):
    probs = []

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(tmpdir)

        # Loop over all extracted images
        for root, _, files in os.walk(tmpdir):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path).convert("RGB")
                    prob = predict_patch(img)
                    probs.append(prob)

    if len(probs) == 0:
        return None, []

    # Aggregate by percentile
    agg_prob = np.percentile(probs, PERCENTILE)
    return agg_prob, probs

# ================================
# Streamlit UI
# ================================
st.title("🧬 BreakHis Slide-Level Classifier")
st.write("Upload a **ZIP file** containing slide patches (e.g., extracted tissue images).")

uploaded_file = st.file_uploader("Upload ZIP of patches", type=["zip"])

if uploaded_file:
    st.info("Running inference on all patches... Please wait ⏳")
    agg_prob, patch_probs = predict_slide_from_zip(uploaded_file)

    if agg_prob is None:
        st.error("❌ No valid image files found in the uploaded ZIP.")
    else:
        # Final slide-level decision
        prediction = "🔴 Malignant" if agg_prob >= THRESHOLD else "🟢 Benign"
        st.subheader("📊 Slide-level Result")
        st.write(f"**{prediction}** (Aggregated Probability = {agg_prob:.2f}, Threshold = {THRESHOLD})")

        # Patch statistics
        st.subheader("📈 Patch-level Summary")
        st.write(f"Number of patches processed: {len(patch_probs)}")
        st.write(f"Patch probabilities → min: {np.min(patch_probs):.2f}, mean: {np.mean(patch_probs):.2f}, max: {np.max(patch_probs):.2f}")

st.subheader("📊 Patch Probability Distribution")
fig, ax = plt.subplots()
ax.hist(patch_probs, bins=10, color="skyblue", edgecolor="black")
ax.set_xlabel("Probability (Malignant)")
ax.set_ylabel("Count")
ax.set_title("Distribution of Patch Predictions")
st.pyplot(fig)
# f"mean: {np.mean(patch_probs):.2f}, max: {np.max(patch_probs):.2f}")
