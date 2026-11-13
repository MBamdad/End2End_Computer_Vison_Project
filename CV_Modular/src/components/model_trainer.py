# Lesson 16: The same simple training loop and test-accuracy calculation.
import torch
import torch.nn as nn
import torch.optim as optim


def train_model(model, train_image_loader, device, epochs=40):
    criterion = nn.CrossEntropyLoss()  # Compare three class scores with labels.
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Update CNN weights.

    # One epoch = one full pass through every training batch.
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_image_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()                # Clear old gradients.
            outputs = model(images)              # Forward pass through CNN.
            loss = criterion(outputs, labels)    # Compute batch error.
            loss.backward()                      # Backpropagation: calculate gradients.
            optimizer.step()                     # Update model weights.
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss : {running_loss/len(train_image_loader)}")


def evaluate_model(model, test_image_loader, device):
    model.eval()  # Switch BatchNorm to evaluation mode.
    correct = 0
    total = 0
    with torch.no_grad():  # No gradients during testing.
        for images, labels in test_image_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Pick the highest-scoring class.
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test accuracy is :{accuracy:.2f}%")
    return accuracy
