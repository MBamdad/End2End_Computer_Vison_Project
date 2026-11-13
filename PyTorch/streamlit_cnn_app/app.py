"""A polished Streamlit interface for the custom 3-class PyTorch CNN.

Only presentation and user interaction have been redesigned. The architecture,
preprocessing, learned weights, and class order are preserved.
"""

from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
import torch
from torch import nn

# The model file lives next to this app in streamlit_cnn_app/model/.
MODEL_PATH = Path(__file__).resolve().parent / "model" / "cnn_128_model-100.pth"
CLASS_NAMES = ["Cat", "Dog", "Person"]
CLASS_ICONS = {"Cat": "🐱", "Dog": "🐶", "Person": "👤"}


class CustomCnnModel(nn.Module):
    """Original CNN architecture and parameter names from the training notebook."""

    def __init__(self):
        super().__init__()
        layers = []
        in_channels = 3
        for out_channels in (32, 64, 128, 256):
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
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


@st.cache_resource(show_spinner="Preparing the CNN model...")
def load_model() -> nn.Module:
    """Load the trusted local model weights once for CPU inference."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model weights were not found at {MODEL_PATH}. "
            "Make sure model/cnn_128_model-100.pth is in the app folder."
        )
    model = CustomCnnModel()
    weights = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model


def predict(image: Image.Image, model: nn.Module) -> tuple[int, np.ndarray]:
    """Preserve the original 128x128 RGB preprocessing and Softmax output."""
    image = image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
    tensor = (tensor - 0.5) / 0.5

    with torch.inference_mode():
        logits = model(tensor)
        scores = torch.softmax(logits, dim=1)[0].numpy()
    return int(np.argmax(scores)), scores


st.set_page_config(
    page_title="VisionLab | CNN Image Classifier",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Minimal, scoped CSS for visual branding; all interactive elements stay native Streamlit.
st.markdown(
    """
    <style>
    .block-container {max-width: 1160px; padding-top: 1.35rem; padding-bottom: 2.2rem;}
    .vl-hero {
        background: linear-gradient(115deg, #122542 0%, #183c55 54%, #115a66 100%);
        color: #ffffff; padding: 2.2rem 2.4rem; border-radius: 22px;
        border: 1px solid rgba(255,255,255,0.12); margin-bottom: 1.2rem;
    }
    .vl-eyebrow {color: #90f0db; font-size: .76rem; font-weight: 750;
        text-transform: uppercase; letter-spacing: .16em; margin-bottom: .55rem;}
    .vl-hero h1 {color: #fff; font-size: clamp(1.9rem, 4vw, 2.7rem); line-height: 1.15;
        margin: .1rem 0 .6rem; font-weight: 770; letter-spacing: -.035em;}
    .vl-hero p {color: #d7e9f0; margin: 0; font-size: 1rem; max-width: 620px;}
    .vl-pills {margin-top: 1.15rem; display: flex; flex-wrap: wrap; gap: .48rem;}
    .vl-pill {display:inline-block; padding:.35rem .7rem; border-radius:99px;
        background: rgba(255,255,255,.105); border:1px solid rgba(255,255,255,.18);
        font-size:.79rem; color:#edf9fa;}
    .vl-section-title {font-size:1.12rem; font-weight:720; margin: 0 0 .25rem;}
    .vl-muted {color:#6b7c8e; font-size:.88rem; margin-bottom: .75rem;}
    .vl-empty {text-align:center; padding: 3.0rem 1rem; border-radius:16px;
        background: linear-gradient(160deg, rgba(38,119,145,.07), rgba(48,96,163,.04));
        border: 1px dashed rgba(84,131,153,.4);}
    .vl-empty-icon {font-size: 2rem; margin-bottom:.35rem;}
    .vl-empty-title {font-size:1.1rem; font-weight:700; margin-bottom:.4rem;}
    .vl-empty-text {color:#6b7c8e; font-size:.86rem;}
    .vl-result {border-radius:16px; padding:1.1rem 1.3rem; margin:.6rem 0 1rem;
        background:linear-gradient(115deg, #e9f8f3, #f1fafb);
        border:1px solid #bee9da; color:#173b32;}
    .vl-result-kicker {text-transform:uppercase; letter-spacing:.11em;
        font-size:.7rem; font-weight:800; color:#347965; margin-bottom:.3rem;}
    .vl-result-label {font-weight:790; font-size:1.7rem; line-height:1.25;}
    .vl-footer {text-align:center; color:#81909e; font-size:.76rem; margin-top:1.3rem;}
    div[data-testid="stButton"] > button[kind="primary"] {border-radius:11px; font-weight:700;}
    @media (max-width:700px) {
       .block-container {padding-top:.8rem;}
       .vl-hero {padding:1.5rem 1.25rem; border-radius:16px;}
       .vl-hero h1 {font-size:1.85rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="vl-hero">
      <div class="vl-eyebrow">Computer Vision · Interactive AI Demo</div>
      <h1>See what your CNN sees.</h1>
      <p>Upload a photo and let a custom PyTorch neural network recognize a
         <b>cat, dog, or person</b> in seconds.</p>
      <div class="vl-pills">
        <span class="vl-pill">⚡ PyTorch inference</span>
        <span class="vl-pill">🧠 Custom CNN</span>
        <span class="vl-pill">☁️ Streamlit Cloud</span>
        <span class="vl-pill">🧪 Educational demo</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.03, 0.97], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="vl-section-title">📤 Upload an image</div>', unsafe_allow_html=True)
        st.caption("Choose a JPG or PNG image to analyze.")
        uploaded_file = st.file_uploader(
            "Choose an image", type=["jpg", "jpeg", "png"],
            label_visibility="collapsed", key="cnn_image_upload",
        )
        uploaded_image = None
        if uploaded_file is not None:
            try:
                uploaded_bytes = uploaded_file.getvalue()
                image_hash = hashlib.sha256(uploaded_bytes).hexdigest()
                if st.session_state.get("last_image_hash") != image_hash:
                    st.session_state["last_image_hash"] = image_hash
                    st.session_state.pop("prediction_result", None)
                uploaded_image = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
                thumbnail = uploaded_image.copy()
                thumbnail.thumbnail((680, 390))  # Show large photos at a tidy preview size.
                st.image(thumbnail, caption="Image preview", width="stretch")
            except Exception:
                st.error("This file could not be opened as an image. Try another JPG or PNG.")
        else:
            st.session_state.pop("last_image_hash", None)
            st.session_state.pop("prediction_result", None)
            st.markdown(
                """<div class="vl-empty">
                  <div class="vl-empty-icon">🖼️</div>
                  <div class="vl-empty-title">Your image goes here</div>
                  <div class="vl-empty-text">Upload a photo to get started.</div>
                </div>""",
                unsafe_allow_html=True,
            )

        clicked = st.button(
            "🔎 Analyze image", type="primary", width="stretch",
            disabled=uploaded_image is None,
        )
        if clicked and uploaded_image is not None:
            try:
                model = load_model()
                started = time.perf_counter()
                class_id, scores = predict(uploaded_image, model)
                elapsed_ms = (time.perf_counter() - started) * 1000
                st.session_state["prediction_result"] = {
                    "class_id": class_id,
                    "scores": scores.tolist(),
                    "elapsed_ms": elapsed_ms,
                }
            except Exception as exc:
                st.session_state.pop("prediction_result", None)
                st.error(f"Unable to analyze the image: {exc}")

with right:
    with st.container(border=True):
        st.markdown('<div class="vl-section-title">✨ Prediction results</div>', unsafe_allow_html=True)
        st.caption("See the predicted category and the model's three Softmax scores.")
        result = st.session_state.get("prediction_result")
        if not result:
            st.markdown(
                """<div class="vl-empty">
                  <div class="vl-empty-icon">🔬</div>
                  <div class="vl-empty-title">Ready to analyze</div>
                  <div class="vl-empty-text">Select an image, then click Analyze image.</div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            label = CLASS_NAMES[result["class_id"]]
            score = float(result["scores"][result["class_id"]])
            st.markdown(
                f"""<div class="vl-result">
                    <div class="vl-result-kicker">Top prediction</div>
                    <div class="vl-result-label">{CLASS_ICONS[label]} {label}</div>
                    </div>""",
                unsafe_allow_html=True,
            )
            m1, m2 = st.columns(2)
            m1.metric("Top Softmax score", f"{score * 100:.1f}%")
            m2.metric("Inference time", f"{result['elapsed_ms']:.0f} ms")
            st.markdown("**Scores by category**")
            for name, probability in zip(CLASS_NAMES, result["scores"]):
                st.write(f"{CLASS_ICONS[name]} **{name}** · {float(probability) * 100:.1f}%")
                st.progress(min(1.0, max(0.0, float(probability))))
            st.caption(
                "Scores are relative to the three trained categories—not guaranteed accuracy. "
                "An unrelated image can still receive a very high score."
            )

st.write("")
with st.expander("ℹ️ How this classifier works"):
    st.write(
        "**Image upload → Resize to 128×128 → Normalize → CNN forward pass "
        "→ Softmax → Predicted category.**"
    )
    st.write(
        "The original PyTorch model uses four convolution blocks and a "
        "fully connected classification head. The trained weights are loaded "
        "once and used on the CPU for prediction."
    )
    st.info(
        "This is a learning project trained to classify Cat, Dog, and Person images. "
        "It is not an industrial defect detector or an open-set classifier."
    )

st.markdown(
    '<div class="vl-footer">Built with PyTorch &amp; Streamlit · '
    'Custom CNN · Educational computer vision project</div>',
    unsafe_allow_html=True,
)
