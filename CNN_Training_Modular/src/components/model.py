# Lesson 16: Original four-block CNN from the training notebook.
import torch
import torch.nn as nn


class CustomCnnModel(nn.Module):
    # __init__: Create the CNN and fully connected layers.
    def __init__(self, input_dim, num_classes):
        super(CustomCnnModel, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.conv_layers = nn.Sequential(
            # C1: 3 -> 32 channels; 128x128 -> 64x64 after MaxPool.
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # C2: 32 -> 64 channels; 64x64 -> 32x32.
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # C3: 64 -> 128 channels; 32x32 -> 16x16.
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # C4: 128 -> 256 channels; 16x16 -> 8x8.
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self._to_linear = None  # Placeholder for the flattened CNN feature count.
        self._get_conv_output(self.input_dim)  # Calculate Flatten size automatically.

        # Fully connected classifier: 16384 -> 512 -> 128 -> 3 scores.
        self.fc_layers = nn.Sequential(
            nn.Linear(self._to_linear, 512),
            nn.ReLU(),
            # nn.Dropout(0.2)  # Commented out in the original lesson.
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes)
        )

    # Use a dummy 128x128 image to find the first Linear layer's input size.
    def _get_conv_output(self, input_dim=128):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_dim, input_dim)
            output = self.conv_layers(dummy_input)
            self._to_linear = output.view(1, -1).size(1)

    # forward: CNN feature extraction -> Flatten -> class scores (logits).
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
