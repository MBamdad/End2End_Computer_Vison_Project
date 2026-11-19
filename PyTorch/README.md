<div align="center">

# 🧠 PyTorch Computer Vision Lab

### From custom CNN training to interactive image-classification applications

**Custom Dataset · PyTorch CNN · Inference Pipeline · Gradio · Streamlit**

![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Task](https://img.shields.io/badge/Task-Image%20Classification-2563EB?style=flat-square)
![Gradio](https://img.shields.io/badge/Interface-Gradio-FF9900?style=flat-square)
![Streamlit](https://img.shields.io/badge/Interface-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Workflow](https://img.shields.io/badge/Workflow-Train%20%E2%86%92%20Deploy-059669?style=flat-square)

**Train a custom three-class CNN → save weights → reuse the model in web interfaces.**

</div>

---

## 🎯 The big picture

This directory explores how to turn a **PyTorch image classifier** into an application that someone else can actually use.

The classification task is **Cat 🐱 / Dog 🐶 / Person 👤**. Rather than treating training and deployment as isolated exercises, the work connects **data preparation**, **network architecture**, **saved model weights**, **inference**, and **user-facing apps**.

## 🗺️ End-to-end learning workflow

```mermaid
flowchart LR
    A["📁 Labeled Images<br/>Custom Dataset"] --> B["🧹 Preprocessing<br/>Resize · Tensor · Normalize"]
    B --> C["🧠 PyTorch CNN<br/>Conv · BN · ReLU · Pool"]
    C --> D["🎯 Training<br/>Loss · Backprop · Optimizer"]
    D --> E[("💾 Model Weights<br/>.pth")]
    E --> F["⚡ Inference<br/>Logits → Softmax"]
    F --> G["🖥️ Gradio / Streamlit<br/>Upload → Prediction"]

    classDef data fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A,stroke-width:2px;
    classDef model fill:#EDE9FE,stroke:#7C3AED,color:#4C1D95,stroke-width:2px;
    classDef train fill:#FEF3C7,stroke:#D97706,color:#92400E,stroke-width:2px;
    classDef saved fill:#D1FAE5,stroke:#059669,color:#065F46,stroke-width:2px;
    classDef app fill:#FCE7F3,stroke:#DB2777,color:#831843,stroke-width:2px;
    class A,B data;
    class C model;
    class D train;
    class E,F saved;
    class G app;
```

> **Key engineering principle:** training is performed separately from serving predictions. Applications reconstruct the same CNN architecture, load the saved `.pth` parameters, switch to evaluation mode, and run inference on uploaded images.

---

## 🧩 Projects in this folder

| Project | Purpose | Does it use a trained CNN? |
| --- | --- | :---: |
| **[01 · Gradio Basics](gradio_app_1/)** | Minimal UI that maps a typed class index to a text label | ❌ |
| **[02 · Gradio CNN Classifier](gradio_app_2/)** | Image upload → PyTorch inference → labeled image | ✅ |
| **[03 · Streamlit CNN Classifier](streamlit_cnn_app/)** | Image upload → prediction + three class scores | ✅ |

### 01 · Gradio Basics — understand the interface

```mermaid
flowchart LR
    A["⌨️ Enter 0, 1 or 2"] --> B["🐍 Python predict()"]
    B --> C["🖥️ Gradio text output"]
    classDef ui fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A,stroke-width:2px;
    classDef logic fill:#FEF3C7,stroke:#D97706,color:#92400E,stroke-width:2px;
    class A,C ui;
    class B logic;
```

**Goal:** learn the link between a Python function and an interactive web input/output. This is deliberately **rule-based**, not neural-network inference.

**Source:** [`gradio_app_1/app.py`](gradio_app_1/app.py)

### 02 · Gradio CNN Classifier — connect a real model

```mermaid
flowchart LR
    A["📤 Upload Image"] --> B["🖼️ RGB · 128×128<br/>Normalize"]
    B --> C["🧠 Trained CNN<br/>.pth"]
    C --> D["📊 Logits → Softmax"]
    D --> E["🏷️ Class / Uncertain<br/>+ Labeled Image"]
    classDef input fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A,stroke-width:2px;
    classDef model fill:#EDE9FE,stroke:#7C3AED,color:#4C1D95,stroke-width:2px;
    classDef result fill:#D1FAE5,stroke:#059669,color:#065F46,stroke-width:2px;
    class A,B input;
    class C,D model;
    class E result;
```

**Goal:** separate UI code from inference. `app.py` accepts the uploaded image while `core/predict.py` loads the trained CNN, processes it and returns the prediction.

The backend uses **four convolutional blocks**, each with **Conv2d → BatchNorm → ReLU → MaxPool**, followed by fully connected classification layers. It draws the predicted label with OpenCV and returns the result to Gradio.

Its current inference code uses **0.80 as a Softmax threshold** to emit `Uncertain` below that score. This is a simple heuristic, **not a reliable out-of-distribution detector**.

**Explore:** [UI](gradio_app_2/app.py) · [Inference backend](gradio_app_2/core/predict.py) · [Project README](gradio_app_2/README.md)

### 03 · Streamlit CNN Classifier — present predictions clearly

```mermaid
flowchart LR
    A["📷 Upload Image"] --> B["⚙️ Preprocess"]
    B --> C["🧠 CNN Inference<br/>CPU"]
    C --> D["🏆 Top Class"]
    C --> E["📈 Softmax Scores<br/>Cat · Dog · Person"]
    classDef input fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A;
    classDef model fill:#EDE9FE,stroke:#7C3AED,color:#4C1D95;
    classDef output fill:#D1FAE5,stroke:#059669,color:#065F46;
    class A,B input;
    class C model;
    class D,E output;
```

**Goal:** demonstrate the same three-class model concept in a richer application: preview uploads, report the most likely class, show all class scores, and measure inference time. The model weights are cached after the initial load.

**Explore:** [Streamlit app](streamlit_cnn_app/app.py) · [Deployment instructions](streamlit_cnn_app/README.md)

---

## 🔬 The CNN architecture — practical overview

```mermaid
flowchart TD
    I["🖼️ RGB Input<br/>3 × 128 × 128"] --> A["Conv Block 1<br/>32 feature channels"]
    A --> B["Conv Block 2<br/>64 feature channels"]
    B --> C["Conv Block 3<br/>128 feature channels"]
    C --> D["Conv Block 4<br/>256 feature channels"]
    D --> E["Flatten<br/>256 × 8 × 8"]
    E --> F["Dense: 512 → 128"]
    F --> G["Output: 3 logits"]
    G --> H["Softmax → Cat / Dog / Person"]
    classDef input fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A;
    classDef conv fill:#EDE9FE,stroke:#7C3AED,color:#4C1D95;
    classDef head fill:#FEF3C7,stroke:#D97706,color:#92400E;
    classDef output fill:#D1FAE5,stroke:#059669,color:#065F46;
    class I input;
    class A,B,C,D conv;
    class E,F,G head;
    class H output;
```

Each block extracts features and halves spatial size through max pooling. The fully connected head combines extracted features into **three class scores**. Softmax converts these logits into relative class probabilities.

**Important:** the classifier predicts **one class for the whole image**. It does **not** locate objects or produce bounding boxes; object detection is covered separately elsewhere in the repository.

---

## 🔁 Training versus inference

| 🏋️ Training | ⚡ Inference |
| --- | --- |
| Labeled images and known targets | A new, unlabeled image |
| Forward pass → loss → backpropagation | Forward pass only |
| Optimizer updates parameters | Parameters stay fixed |
| Save learned parameters to `.pth` | Load the trained `.pth` |
| Typically `model.train()` | `model.eval()` and `torch.no_grad()` / `torch.inference_mode()` |

```mermaid
flowchart LR
    A["📷 Input + Ground Truth"] --> B["🧠 Model"]
    B --> C["📉 Compare Prediction<br/>with True Label"]
    C --> D["🔄 Backpropagation<br/>+ Optimizer"]
    D --> B
    B --> E[("💾 Save .pth")]
    E --> F["📤 New Input"]
    F --> G["🏷️ Prediction"]
    classDef basic fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A;
    classDef learn fill:#FEF3C7,stroke:#D97706,color:#92400E;
    classDef output fill:#D1FAE5,stroke:#059669,color:#065F46;
    class A,B basic;
    class C,D learn;
    class E,F,G output;
```

The deployed Gradio and Streamlit apps perform **inference**, not retraining or transfer learning.

---

## 🗂️ Repository map

```text
PyTorch/
├── README.md                      # This guide
├── gradio_app_1/
│   ├── app.py                     # Rule-based Gradio function demo
│   ├── requirements.txt
│   └── README.md
├── gradio_app_2/
│   ├── app.py                     # Gradio image input / output
│   ├── core/
│   │   └── predict.py            # CNN architecture, weights, inference
│   ├── model/                     # Trained model weights
│   ├── requirements.txt
│   └── README.md
└── streamlit_cnn_app/
    ├── app.py                     # Streamlit interface and inference
    ├── model/                     # Trained model weights
    ├── requirements.txt
    └── README.md
```

*The custom-dataset training notebook is intentionally kept locally rather than tracked in this remote directory.*

## 🚀 Try the apps locally

Run commands from the corresponding app directory. Each app maintains its own dependencies.

| App | Install | Start |
| --- | --- | --- |
| **Gradio basic** | `pip install -r requirements.txt` | `python app.py` |
| **Gradio CNN** | `pip install -r requirements.txt` | `python app.py` |
| **Streamlit CNN** | `pip install -r requirements.txt` | `streamlit run app.py` |

For the Streamlit app, you can alternatively launch from the repository root:

```bash
python -m pip install -r PyTorch/streamlit_cnn_app/requirements.txt
python -m streamlit run PyTorch/streamlit_cnn_app/app.py
```

Keep `cnn_128_model-100.pth` in the application-specific `model/` directory. The Gradio CNN backend automatically selects CUDA if available, otherwise CPU. The Streamlit implementation explicitly loads weights on CPU.

## 🧪 Scope and engineering considerations

- **Demonstrated:** custom CNN architecture, reuse of saved training weights, inference preprocessing, modular prediction code, Gradio and Streamlit user interfaces.
- **Not claimed:** industrial-grade generalization, calibrated probability estimates, robustness to arbitrary out-of-class images, production monitoring, or automated test coverage.
- **Model limitation:** every image is assigned one of three known classes by default; a high Softmax score does **not** prove the image belongs to any of them.
- **Gradio threshold:** the optional `Uncertain` outcome reduces some low-confidence predictions but cannot reliably solve open-set recognition.

---

<div align="center">

### 💡 The engineering takeaway

**A neural network becomes useful when its learned weights can be loaded, evaluated, and integrated into a repeatable user workflow.**

[⬅️ Back to the main Computer Vision repository](../README.md)

</div>
