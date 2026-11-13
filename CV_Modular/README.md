# End-to-End Computer Vision | Modular PyTorch CNN

**A small, beginner-friendly, end-to-end image-classification project built from lessons 15–16.**

This is deliberately **simple**: the original CNN blocks, loss function, Adam optimizer, training loop, evaluation, and OpenCV prediction logic are kept as close to the course as possible. The only goal of modularization is to put each part in the right file and learn how the pieces connect.

> **Start here:** Open [`notebook/01_CNN_Modular_Tutorial.ipynb`](notebook/01_CNN_Modular_Tutorial.ipynb) in **VS Code**. It is a step-by-step walkthrough with Markdown explanations, code and short `#` reminders. You do not need to learn any new framework.

## 🎯 What does the model do?

Predict one of **three classes** from an RGB image: **Cat**, **Dog** or **person**.

```text
Images organized in folders
             │
             ▼
    ImageDataset (one image + label)
             │
             ▼
    DataLoader (32 images per batch)
             │
             ▼
      CNN (4 feature blocks)
             │
             ▼
    Flatten → Linear → 3 class scores
             │
             ▼
 Training → Evaluation → Save .pth
             │
             ▼
    Inference → OpenCV label on image
```

## 📁 Simple project structure

```text
Computer_Vision/
├── PyTorch/                          # Your existing lessons + image dataset (unchanged)
│   └── Classification_dataset_v3/
│       └── images/{train,test}/{Cat,Dog,person}/
├── requirements.txt                  # Your existing dependencies (unchanged)
└── CV_Modular/                       # This new teaching project
    ├── notebook/
    │   └── 01_CNN_Modular_Tutorial.ipynb  # Start here: Markdown + executable code
    ├── src/
    │   ├── components/
    │   │   ├── data_ingestion.py      # Dataset: one image and its label
    │   │   ├── data_transformation.py # Resize, ToTensor, Normalize
    │   │   ├── model.py               # Original four-block CNN
    │   │   └── model_trainer.py       # Training loop + test accuracy
    │   └── pipeline/
    │       ├── train_pipeline.py      # Connect modules and save model
    │       └── predict_pipeline.py    # Load weights and predict new image
    ├── artifacts/                     # Generated locally, never committed
    ├── setup.py                       # Optional package metadata
    ├── .gitignore
    └── README.md
```

**Why `.py` as well as `.ipynb`?** The `.ipynb` is your **teaching notebook**, with friendly Markdown explanations and experiments. The small `.py` files are **reusable modules** that the notebook calls. Both are ordinary VS Code files; `.py` does **not** mean PyCharm.

## 🧠 Remember the core ideas

| Code / component | One-line memory tip |
|---|---|
| `Dataset` | How to read and prepare **one** image and label. |
| `__init__` | Collect file paths and class labels. |
| `__len__` | Count images. |
| `__getitem__` | Load and transform the image at one index. |
| `DataLoader` | Group samples into batches; optionally shuffle. |
| `transforms.Resize((128, 128))` | Give each image the same spatial dimensions. |
| `ToTensor()` | Convert image to a PyTorch `(C, H, W)` tensor. |
| `Normalize(0.5, 0.5)` | Apply the same normalization during training **and** prediction. |
| `conv_layers` | Extract useful features with Conv → BN → ReLU → MaxPool. |
| `_get_conv_output` | Use a dummy image to determine the flattened feature count. |
| `fc_layers` | Turn image features into 3 class scores. |
| `forward()` | Tell PyTorch how input passes through the CNN. |
| `loss.backward()` | Calculate gradients via backpropagation. |
| `optimizer.step()` | Update model weights. |
| `model.eval()` | Switch model to evaluation / prediction behavior. |
| `torch.save(model.state_dict(), ...)` | Save learned weights to `.pth`. |
| `cv2.putText` | Write the predicted class name onto the output image. |

### 🔢 Tensor shape roadmap

| Stage | Shape (`batch_size=32`) |
|---|---|
| Input RGB images | `(32, 3, 128, 128)` |
| Block 1 | `(32, 32, 64, 64)` |
| Block 2 | `(32, 64, 32, 32)` |
| Block 3 | `(32, 128, 16, 16)` |
| Block 4 | `(32, 256, 8, 8)` |
| Flatten | `(32, 16384)` |
| Linear layers | `16384 → 512 → 128 → 3` |
| Output | `(32, 3)` raw class scores (logits) |

The first `3` comes from RGB; `32, 64, 128, 256` are **chosen output-channel hyperparameters**. Four `2×2` MaxPools reduce `128 → 64 → 32 → 16 → 8`. The same `Resize(128,128)` and `input_dim=128` must be used together for this architecture.

## ▶️ Run locally in VS Code (Windows / PowerShell)

1. Open `C:\Job\ML_Data_Scientist\IAB\Computer_Vision` in VS Code. Select the Python environment with PyTorch installed.
2. Keep your existing dataset in `Computer_Vision\PyTorch\Classification_dataset_v3`.
3. Open `CV_Modular/notebook/01_CNN_Modular_Tutorial.ipynb` and run cells **in order**.

Or run the same modular pipeline from a VS Code terminal:

```powershell
cd "C:\Job\ML_Data_Scientist\IAB\Computer_Vision\CV_Modular"

# Only if your environment needs the existing dependencies:
pip install -r ..\requirements.txt

# Quick learning run first, then the tutorial's full 40 epochs:
python -m src.pipeline.train_pipeline --epochs 1
python -m src.pipeline.train_pipeline --epochs 40

# Predict one image after training:
python -m src.pipeline.predict_pipeline "..\PyTorch\Classification_dataset_v3\images\test\Dog\dog_1000.jpg"
```

The saved weights and annotated image will appear inside `CV_Modular/artifacts/`. This generated folder is excluded from Git.

**Dataset not found?** This package expects `Computer_Vision/PyTorch/Classification_dataset_v3/images/train` and `images/test`. If your folder has another name, change only `DATASET_DIR` in `src/pipeline/train_pipeline.py`.

## 📚 Dataset and lesson references

- Dataset: [Classification_dataset_v3 — Google Drive](https://drive.google.com/file/d/1G9H2W0R6JLYYBXyHNM_kAcBufUJr4Qsy/view?usp=sharing)
- Udemy lessons: **Defining custom Image Dataset loader and usage**; **CNN Training Using a Custom Dataset**.
- Original `015` and `016` course notebooks are kept in your existing `PyTorch` directory, not overwritten here.

## 🛠️ Original code vs. tiny practical adjustments

The 4 CNN blocks, fully connected layers, dummy-input size calculation, `CrossEntropyLoss`, Adam `lr=0.001`, loss/backward/step loop, accuracy formula, saving with `.pth`, and image prediction steps follow the course examples.

Only a few practical adjustments were needed to run as a reusable local project:
- `sorted(os.listdir(image_dir))` ensures the **same class numbers** for train/test (`Cat=0`, `Dog=1`, `person=2`).
- Folder paths use your Windows project structure rather than Google Colab `/content/` paths.
- `shuffle=False` for the test loader; the training loader remains `shuffle=True`.
- The **single shared transform** is used for training and inference, including `Normalize(mean=[0.5]*3, std=[0.5]*3)` as in your current lesson 016 notebook.
- `torch.load(..., weights_only=True)` loads trusted model state dictionaries more safely.

## 💼 Why this matters for industrial AI

The workflow is a reusable **classification template**: replace the three-class training images with labeled industrial images (e.g., healthy vs defective components), then train and evaluate a model on held-out data. Production use would additionally require stronger validation, appropriate datasets, monitoring, and deployment—not included in this educational example.

**Scope:** portfolio learning project; no industrial performance claims. CNN accuracy depends on the dataset, training time, hardware, and random seed, so run your own test rather than relying on a fixed number.
