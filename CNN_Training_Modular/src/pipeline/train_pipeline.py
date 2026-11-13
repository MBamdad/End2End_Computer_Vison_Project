"""Run from CV_Modular: python -m src.pipeline.train_pipeline --epochs 40"""
from pathlib import Path
import argparse
import torch
from torch.utils.data import DataLoader

from src.components.data_ingestion import ImageDataset
from src.components.data_transformation import transform
from src.components.model import CustomCnnModel
from src.components.model_trainer import train_model, evaluate_model

# Your existing dataset remains in Computer_Vision/PyTorch/ (do not copy it).
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_DIR.parent / "PyTorch" / "Classification_dataset_v3"
MODEL_PATH = PROJECT_DIR / "artifacts" / "cnn_model.pth"


def main(epochs=40):
    train_image_dir = DATASET_DIR / "images" / "train"
    test_image_dir = DATASET_DIR / "images" / "test"

    # Custom Dataset = prepare ONE sample; DataLoader = group 32 samples.
    train_image_dataset = ImageDataset(str(train_image_dir), transform=transform)
    test_image_dataset = ImageDataset(str(test_image_dir), transform=transform)
    train_image_loader = DataLoader(dataset=train_image_dataset, batch_size=32, shuffle=True)
    test_image_loader = DataLoader(dataset=test_image_dataset, batch_size=32, shuffle=False)

    print("Class names:", train_image_dataset.class_name)
    print("Batches per epoch:", len(train_image_loader))

    # Same CPU/GPU choice and same CNN as in the tutorial.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomCnnModel(input_dim=128, num_classes=3).to(device)

    train_model(model, train_image_loader, device, epochs=epochs)
    evaluate_model(model, test_image_loader, device)

    # Save weights only, using the familiar .pth format.
    MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), str(MODEL_PATH))
    print("Saved model:", MODEL_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    main(epochs=args.epochs)
