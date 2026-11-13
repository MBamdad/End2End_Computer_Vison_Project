"""Run from CV_Modular: python -m src.pipeline.predict_pipeline PATH_TO_IMAGE"""
import sys
from pathlib import Path
import cv2
import torch
from PIL import Image

from src.components.data_ingestion import ImageDataset
from src.components.data_transformation import transform
from src.components.model import CustomCnnModel
from src.pipeline.train_pipeline import DATASET_DIR, MODEL_PATH, PROJECT_DIR


class ImageClassifier:
    # __init__: Build CNN once, then load its previously trained weights.
    def __init__(self, model_path, class_names):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CustomCnnModel(input_dim=128, num_classes=3).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        self.class_names = class_names
        self.transform = transform  # Exactly the same Resize/ToTensor/Normalize as training.

    # predict: Classify ONE new picture; write its class name on a copy.
    def predict(self, image_path, output_path="output_image.jpg"):
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            _, predicted = torch.max(output, 1)
        label = self.class_names[predicted.item()]

        img = cv2.imread(str(image_path))  # OpenCV draws the label; CNN predicts it.
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imwrite(str(output_path), img)
        return label


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m src.pipeline.predict_pipeline "path/to/image.jpg"')
    image_path = sys.argv[1]
    train_image_dataset = ImageDataset(str(DATASET_DIR / "images" / "train"))
    classifier = ImageClassifier(str(MODEL_PATH), train_image_dataset.class_name)
    output_path = PROJECT_DIR / "artifacts" / "output_image.jpg"
    output_path.parent.mkdir(exist_ok=True)
    label = classifier.predict(image_path, output_path)
    print(f"Predicted class is : {label}")
    print("Annotated image:", output_path)
