# Gradio App 2 — CNN Image Classification

A PyTorch and Gradio application for classifying uploaded images as **Cat**, **Dog**, or **Person** using a pretrained custom CNN.

## How it works

1. Upload an image through the Gradio interface.
2. The backend loads and preprocesses the image, then runs CNN inference.
3. The app displays the predicted class and an annotated image.

**Project files:** `app.py` (Gradio UI), `core/predict.py` (model loading and inference), and `model/` (trained weights).

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open the local URL shown in the terminal. Keep the trained model weights in `model/`.
