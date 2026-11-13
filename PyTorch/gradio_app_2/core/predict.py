import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os


# Define the same CNN architecture used during training
class CustomCnnModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CustomCnnModel, self).__init__()  # Initialize PyTorch model
        self.input_dim = input_dim  # Input image size (128)
        self.num_classes = num_classes  # Number of classes (3)

        # Extract image features using convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # RGB → 32 feature maps
            nn.BatchNorm2d(32),  # Normalize feature maps
            nn.ReLU(),  # Apply nonlinear activation
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce image dimensions by half

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 32 → 64 feature maps
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 64 → 128 feature maps
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 128 → 256 feature maps
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Compute dynamically the input size for fully connected layers
        self._to_linear = None  # Placeholder for flattened feature size
        self._get_conv_output(self.input_dim)  # Calculate feature size automatically

        # Fully connected layers for final classification
        self.fc_layers = nn.Sequential(
            nn.Linear(self._to_linear, 512),  # Features → 512 neurons
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(512, 128),  # 512 → 128 neurons
            nn.ReLU(),
            nn.Linear(128, self.num_classes)  # 128 → 3 class scores
        )

    def _get_conv_output(self, input_dim=224):
      with torch.no_grad():  # Disable gradient tracking
          dummy_input = torch.zeros(1, 3, input_dim, input_dim)  # Create a dummy RGB image
          output = self.conv_layers(dummy_input)  # Pass through CNN layers
          self._to_linear = output.view(1, -1).size(1)  # Find flattened feature count

    def forward(self, x):  # Define how input flows through the CNN
      x = self.conv_layers(x)  # Extract image features
      x = x.view(x.size(0), -1)  # Flatten features, keeping batch dimension
      x = self.fc_layers(x)  # Generate classification scores
      return x  # Return the scores for each class

# Load the trained model and predict classes for new images
class ImageClassifier():
    def __init__(self, model_path, class_name=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise CPU
        self.model = CustomCnnModel(input_dim=128, num_classes=3).to(self.device)  # Create CNN on selected device
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))  # Load saved trained weights
        self.model.eval()  # Switch model to evaluation mode

        if class_name is None:  # Use default class names
            self.class_name = {0: 'Cat', 1: 'Dog', 2: 'person'}  # Map class indices to labels
        else:
            self.class_name = class_name  # Use custom class names

        # Apply the same preprocessing used during training
        self.transform = transforms.Compose([
        transforms.Resize((128,128)),  # Resize image to 128×128
        transforms.ToTensor(),  # Convert image to tensor (C, H, W)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]  # Normalize pixels to [-1, 1]
        )

    def predict(self, image_path):  # Predict the class of a new image
        image = Image.open(image_path).convert("RGB")  # Load image in RGB format
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)  # Preprocess + add batch dimension + move to device

        with torch.no_grad():
            output = self.model(image_tensor)  # Get raw CNN scores (logits)
            probabilities = torch.softmax(output, dim=1)  # Convert logits into class probabilities
            print("Class probabilities:", probabilities[0].tolist(), flush=True)
            confidence, predicted = torch.max(probabilities, dim=1)  # Get highest probability and class index

        # Accept prediction only if confidence is at least 80%
        if confidence.item() < 0.80:
            label = "Uncertain"  # Model is not confident enough
        else:
            label = self.class_name[predicted.item()]  # Return predicted class

        img = cv2.imread(image_path)  # Load original image with OpenCV
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # Draw blue class label on image
        output_path = "labeled_image.jpg"  # Name of output image
        cv2.imwrite(output_path, img)  # Save labeled image
        cwd = os.getcwd()  # Get current working directory
        output_path = os.path.join(cwd, output_path)  # Create full output image path

        return label, output_path  # Return predicted class and labeled image path