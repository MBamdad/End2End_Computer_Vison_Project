# Lesson 15: One image + one true label comes from Dataset.
import os
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    # Collect image paths and corresponding class labels.
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = []
        self.labels = []
        self.class_name = {}
        self.transform = transform

        # sorted() keeps Cat=0, Dog=1, person=2 in BOTH train and test.
        for label, class_dir in enumerate(sorted(os.listdir(image_dir))):
            self.class_name[label] = class_dir
            class_path = os.path.join(image_dir, class_dir)
            for img_name in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, img_name))
                self.labels.append(label)

    # Return the total number of images.
    def __len__(self):
        return len(self.image_paths)

    # Read and prepare ONE image with its label.
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
