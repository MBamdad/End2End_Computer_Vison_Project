# End-to-End Computer Vision | Modular CNN Image Classification

A modular image-classification project built with **PyTorch**. The pipeline covers image preprocessing, CNN training, model evaluation, checkpoint saving, and prediction on new images. The code is organized into reusable components while keeping the implementation concise and easy to extend.

## Project Overview

| Property | Specification |
|---|---|
| **Task** | Multi-class image classification |
| **Classes** | Cat, Dog, person |
| **Input** | RGB images resized to 128 × 128 pixels |
| **Model** | Custom convolutional neural network (CNN) |
| **Frameworks** | PyTorch, Torchvision, OpenCV |
| **Training** | Cross-entropy loss and Adam optimizer |
| **Outputs** | Test accuracy, saved model weights, annotated predictions |

## Pipeline

```mermaid
flowchart LR
    A["Labeled images"] --> B["Custom Dataset"]
    B --> C["Transforms & DataLoader"]
    C --> D["CNN feature extraction"]
    D --> E["Fully connected classifier"]
    E --> F["Training & evaluation"]
    F --> G["Saved weights (.pth)"]
    G --> H["Inference on new images"]
```

**Data preparation:** `ImageDataset` reads an image and its label. The `DataLoader` groups samples into batches of 32 and shuffles training data. Images are resized, converted to tensors, and normalized consistently for training and inference.

**Model training:** Four convolutional blocks extract image features. Fully connected layers convert those features into three class scores. The model learns through backpropagation and Adam weight updates.

**Evaluation and inference:** Accuracy is measured using the held-out test set. The trained weights are saved as a `.pth` file and reloaded for predictions. OpenCV adds the predicted class name to an output image.

## CNN Architecture

Each convolutional block follows:

`Conv2D → BatchNorm2D → ReLU → MaxPool2D`

| Stage | Output shape per image |
|---|---|
| RGB input | `3 × 128 × 128` |
| CNN block 1 | `32 × 64 × 64` |
| CNN block 2 | `64 × 32 × 32` |
| CNN block 3 | `128 × 16 × 16` |
| CNN block 4 | `256 × 8 × 8` |
| Flatten | `16,384` features |
| Fully connected layers | `16,384 → 512 → 128 → 3` |

The final layer produces **three logits** (raw class scores). Convolution output channels (`32`, `64`, `128`, `256`) are architectural choices; spatial dimensions decrease through max-pooling.

## Project Structure

```text
CV_Modular/
├── notebook/                    # Exploration and visual checks
├── src/
│   ├── components/
│   │   ├── data_ingestion.py     # ImageDataset: image paths and class labels
│   │   ├── data_transformation.py # Resize, tensor conversion, normalization
│   │   ├── model.py              # CNN architecture and forward pass
│   │   └── model_trainer.py      # Training loop and test accuracy
│   └── pipeline/
│       ├── train_pipeline.py     # Train, evaluate, and save model
│       └── predict_pipeline.py   # Load weights and classify images
├── setup.py                     # Package metadata
└── README.md
```

The notebooks support exploration and inspection; the Python modules provide a reusable training and inference workflow.

## Dataset

**Dataset:** [Classification Dataset (Google Drive)](https://drive.google.com/file/d/1G9H2W0R6JLYYBXyHNM_kAcBufUJr4Qsy/view?usp=sharing)

Download and extract the dataset into the following location **relative to the repository root**:

```text
PyTorch/
└── Classification_dataset_v3/
    └── images/
        ├── train/
        │   ├── Cat/
        │   ├── Dog/
        │   └── person/
        └── test/
            ├── Cat/
            ├── Dog/
            └── person/
```

Class indices are assigned consistently from sorted folder names: `Cat = 0`, `Dog = 1`, `person = 2`. The image dataset is not stored in this repository.

## Installation and Usage

From the **repository root**, enter the project directory and install the existing dependencies:

```bash
cd CV_Modular
python -m pip install -r ../requirements.txt
```

### Train and evaluate

```bash
python -m src.pipeline.train_pipeline --epochs 40
```

The pipeline loads the training and test datasets, trains the CNN, reports test accuracy, and saves its weights to:

```text
CV_Modular/artifacts/cnn_model.pth
```

### Predict a new image

After training, run:

```bash
python -m src.pipeline.predict_pipeline "../PyTorch/Classification_dataset_v3/images/test/Dog/dog_1000.jpg"
```

The predicted class is printed in the terminal. An annotated copy of the image is saved to:

```text
CV_Modular/artifacts/output_image.jpg
```

Replace the example path with any supported image you want to classify. The model predicts from the **image pixels**, not from the folder or filename.

## Implementation Notes

- **Preprocessing:** `Resize((128, 128))`, `ToTensor()`, and `Normalize(mean=[0.5]*3, std=[0.5]*3)` are used consistently for training and inference.
- **Batching:** The training DataLoader uses `batch_size=32` and `shuffle=True`; the test DataLoader uses `shuffle=False`.
- **Training:** The network uses `CrossEntropyLoss` and Adam with a learning rate of `0.001`. The default training run is 40 epochs.
- **Model persistence:** `torch.save(model.state_dict(), ...)` stores model parameters and registered buffers. Inference recreates the CNN architecture and loads these saved values.
- **Evaluation:** Test accuracy is the proportion of correctly classified images. Results depend on the training run and should be measured rather than assumed.

## Potential Applications

This pipeline provides a starting point for image-based classification tasks such as visual quality inspection and component categorization. Adapting it to a new domain requires relevant labeled data, task-specific training, and validation on representative unseen images.