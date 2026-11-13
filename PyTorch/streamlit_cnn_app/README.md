# CNN Image Classification — Streamlit

A lightweight Streamlit application for the custom PyTorch CNN from the Computer Vision course. Upload a JPG or PNG image to predict **Cat**, **Dog**, or **Person** and view class probabilities.

## Model

Uses the existing trained weights at `../gradio_app_2/model/cnn_128_model-100.pth`. No retraining or duplicate model upload is required.

## Run locally

From the Git repository root:

```bash
pip install -r PyTorch/streamlit_cnn_app/requirements.txt
streamlit run PyTorch/streamlit_cnn_app/app.py
```

## Deploy

On [Streamlit Community Cloud](https://share.streamlit.io/), choose your GitHub repository and `main` branch, with the entrypoint `PyTorch/streamlit_cnn_app/app.py`. Keep the sibling `gradio_app_2/model` directory in the same repository.

Educational classification demo; not a validated industrial-quality inspection model.
