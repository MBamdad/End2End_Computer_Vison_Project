"""Streamlit UI for the trained 3-class CNN from the PyTorch course."""
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
import torch
from torch import nn

# Reuse the existing model weights already stored in our GitHub project.
MODEL_PATH = Path(__file__).resolve().parent / "model" / "cnn_128_model-100.pth"
CLASS_NAMES = ["Cat", "Dog", "Person"]

class CustomCnnModel(nn.Module):
    """Same architecture and layer names as the model trained in lesson 119."""

    def __init__(self):
        super().__init__()
        layers = []
        in_channels = 3
        for out_channels in (32, 64, 128, 256):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ])
            in_channels = out_channels
        self.conv_layers = nn.Sequential(*layers)
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, len(CLASS_NAMES)),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc_layers(x)


@st.cache_resource(show_spinner="Loading trained CNN...")
def load_model():
    """Load once and reuse across Streamlit reruns (CPU inference)."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing trained model: {MODEL_PATH}")
    model = CustomCnnModel()
    weights = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model


def predict(image: Image.Image, model: nn.Module):
    """Apply the same Resize/ToTensor/Normalize steps used in training."""
    image = image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
    tensor = (tensor - 0.5) / 0.5

    with torch.inference_mode():
        logits = model(tensor)
        scores = torch.softmax(logits, dim=1)[0].numpy()
    return int(np.argmax(scores)), scores


st.set_page_config(page_title="CNN Image Classifier", page_icon="🔍", layout="centered")
st.title("🔍 CNN Image Classification")
st.write("Upload a photo and classify it as **Cat**, **Dog**, or **Person** using a custom PyTorch CNN.")
st.caption("Educational model from the Computer Vision course — not an industrial defect detector.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        uploaded_image = Image.open(uploaded_file).convert("RGB")
        st.image(uploaded_image, caption="Uploaded image", use_container_width=True)
        if st.button("Classify Image", type="primary"):
            model = load_model()
            class_id, probabilities = predict(uploaded_image, model)
            st.success(f"Prediction: **{CLASS_NAMES[class_id]}**")
            st.metric("Model confidence", f"{probabilities[class_id] * 100:.1f}%")
            st.write("**Predicted probabilities**")
            for label, probability in zip(CLASS_NAMES, probabilities):
                st.write(f"{label}: {probability * 100:.1f}%")
                st.progress(float(probability))
    except Exception as exc:
        st.error(f"The image could not be classified: {exc}")
else:
    st.info("Upload a JPG or PNG image to start.")
