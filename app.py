import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json, os

# --- CONFIG ---
MODEL_PATH = "unfreeze_2_layers_finetuned_acc0.962.pth"
THRESHOLD_PATH = "unfreeze_2_layers_finetuned_acc0.962_slide_threshold.json"
DEVICE = torch.device("cpu")

# --- Load Model & Threshold ---
@st.cache_resource
def load_model_and_threshold(model_path, threshold_path):
    # Load model
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # Load threshold
    threshold = 0.5
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            data = json.load(f)
        threshold = data.get("best_threshold", threshold)  # ✅ use best_threshold

    return model, threshold

# --- Preprocessing ---
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# --- Inference ---
def predict(model, image, threshold):
    x = preprocess_image(image).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    benign_prob, malignant_prob = probs.tolist()
    prediction = "Malignant" if malignant_prob >= threshold else "Benign"
    return prediction, benign_prob, malignant_prob

# --- Streamlit UI ---
st.set_page_config(page_title="BreakHis Classifier", page_icon="🧬")

st.title("🧬 BreakHis Classifier")
st.write("Upload a histopathology image to classify as **Benign** or **Malignant**.")

model, best_threshold = load_model_and_threshold(MODEL_PATH, THRESHOLD_PATH)

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    prediction, benign_prob, malignant_prob = predict(model, image, best_threshold)

    st.markdown(f"**Prediction:** {prediction}")
    st.markdown(f"**Benign probability:** {benign_prob:.2f}")
    st.markdown(f"**Malignant probability:** {malignant_prob:.2f}")
