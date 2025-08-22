import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os, zipfile, tempfile
import matplotlib.pyplot as plt

# =====================
# Configurations
# =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_resnet50_balanced.pth"
THRESHOLD = 0.39  # best threshold from eval

# =====================
# Model Definition
# =====================
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =====================
# Transform
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =====================
# Streamlit App
# =====================
st.title("🧬 Histopathology Slide Classifier")
st.write("Upload either a single image or a ZIP folder of patches.")

uploaded_file = st.file_uploader("Upload an image or ZIP file", type=["jpg", "jpeg", "png", "zip"])

if uploaded_file:
    if uploaded_file.name.endswith((".jpg", ".jpeg", ".png")):
        # ===== Single Image =====
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = torch.softmax(model(img_tensor), dim=1)[0, 1].item()

        pred_class = int(prob >= THRESHOLD)
        st.markdown(
            f"Result: {'🔴 Malignant' if pred_class else '🟢 Benign'} "
            f"(Probability = {prob:.2f}, Threshold = {THRESHOLD:.2f})"
        )

    elif uploaded_file.name.endswith(".zip"):
        # ===== ZIP of patches =====
        status_placeholder = st.empty()
        status_placeholder.info("📂 Extracting ZIP patches...")

        tmpdir = tempfile.mkdtemp()
        with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        # Collect images (recursive)
        patch_paths = []
        for root, _, files in os.walk(tmpdir):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    patch_paths.append(os.path.join(root, file))

        if not patch_paths:
            st.error("⚠️ No image patches found inside the ZIP file. Please check file format.")
            st.write("Files found in ZIP:")
            st.write(os.listdir(tmpdir))
        else:
            status_placeholder.info(f"Running inference on {len(patch_paths)} patches... Please wait ⏳")
            patch_probs = []

            for p in patch_paths:
                img = Image.open(p).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    prob = torch.softmax(model(img_tensor), dim=1)[0, 1].item()
                patch_probs.append(prob)

            # Slide-level aggregation
            slide_prob = np.mean(patch_probs)
            pred_class = int(slide_prob >= THRESHOLD)

            st.subheader("📊 Slide-level Result")
            st.markdown(
                f"{'🔴 Malignant' if pred_class else '🟢 Benign'} "
                f"(Aggregated Probability = {slide_prob:.2f}, Threshold = {THRESHOLD:.2f})"
            )

            st.subheader("📈 Patch-level Summary")
            st.write(f"Number of patches processed: {len(patch_probs)}")
            st.write(
                f"Patch probabilities → min: {np.min(patch_probs):.2f}, "
                f"mean: {np.mean(patch_probs):.2f}, max: {np.max(patch_probs):.2f}"
            )

            # Histogram
            fig, ax = plt.subplots()
            ax.hist(patch_probs, bins=10, color="skyblue", edgecolor="black")
            ax.set_title("Distribution of Patch Probabilities")
            ax.set_xlabel("Probability of Malignant")
            ax.set_ylabel("Count")
            st.pyplot(fig)

            status_placeholder.empty()