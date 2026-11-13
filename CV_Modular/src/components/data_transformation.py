# Lesson 15: Apply the SAME preprocessing during training and prediction.
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Every image becomes 128x128.
    transforms.ToTensor(),           # RGB image -> (3, 128, 128) Tensor.
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
