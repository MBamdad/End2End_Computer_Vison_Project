# CNN Image Classification — Streamlit App

An interactive image-classification demo built with **PyTorch** and **Streamlit**. Upload an image to classify it as **Cat**, **Dog**, or **Person** using a custom-trained CNN.

The app displays the predicted class and the model's Softmax scores for all three classes.

## Project Structure

```text
streamlit_cnn_app/
├── app.py                         # Streamlit interface and CNN inference
├── model/
│   └── cnn_128_model-100.pth      # Trained model weights
├── requirements.txt               # Python dependencies
└── README.md
```

## Run Locally

Use **Python 3.11**. From the root of the GitHub repository, run:

```bash
python -m pip install -r PyTorch/streamlit_cnn_app/requirements.txt
python -m streamlit run PyTorch/streamlit_cnn_app/app.py
```

Open the local URL displayed in the terminal (typically `http://localhost:8501`).

## Free Deployment with Streamlit Community Cloud

[Streamlit Community Cloud](https://share.streamlit.io/) is a **free hosting platform** for Streamlit apps. It connects to GitHub, installs dependencies from `requirements.txt`, runs the Python application on a cloud server, and provides a shareable `*.streamlit.app` URL—without needing to manage a server yourself. Free hosting has resource limits.

**To deploy this project:**

1. Push `streamlit_cnn_app/` (including the `.pth` model file) to your GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub and authorize repository access.
3. Select **Create app → Deploy a public app from GitHub**.
4. Enter your repository, select branch **`main`**, and set the **main file path** to `PyTorch/streamlit_cnn_app/app.py`.
5. In **Advanced settings**, choose **Python 3.11**, then click **Deploy**.
6. Open the generated URL and test the classifier with an image.

Both public and private GitHub repositories are supported. If your repository is private, check the app's **sharing settings** before sending the link to others. Pushing changes to the deployed GitHub branch normally updates the cloud app automatically.

## Note

This is an **educational three-class classifier**, not a validated industrial inspection system. Softmax scores are not guaranteed accuracy estimates, and images outside the three classes can still receive confident predictions.
