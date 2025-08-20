import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json

# ================================
# Settings
# ================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "deploy_resnet50_slide.pth"
THRESHOLD_PATH = "deploy_resnet50_slide_threshold.json"

# ================================
# Model Loader
# ================================
@st.cache_resource
def load_model_and_threshold(model_path, threshold_path):
    # Load model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)  # 2 classes: benign, malignant

    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    # Load tuned threshold
    with open(threshold_path, "r") as f:
        threshold = json.load(f).get("best_threshold", 0.5)

    return model, threshold

model, best_threshold = load_model_and_threshold(MODEL_PATH, THRESHOLD_PATH)

# ================================
# Transform for Input Images
# ================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ================================
# Prediction Function
# ================================
def predict(image: Image.Image):
    img_t = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    malignant_prob = probs[1]
    pred_class = 1 if malignant_prob >= best_threshold else 0
    return pred_class, malignant_prob

# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="BreakHis Classifier", page_icon="🧬")

st.title("🧬 BreakHis Histopathology Classifier")
st.write("Upload a histopathology patch image to classify as **Benign** or **Malignant**.")

# ✅ Status line to confirm tuned threshold
st.info(f"🔧 Using tuned decision threshold = **{best_threshold:.2f}**")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    pred_class, malignant_prob = predict(image)

    st.subheader("Prediction")
    if pred_class == 1:
        st.error(f"🔴 Malignant (Probability = {malignant_prob:.2f}, Threshold = {best_threshold:.2f})")
    else:
        st.success(f"🟢 Benign (Probability = {malignant_prob:.2f}, Threshold = {best_threshold:.2f})")