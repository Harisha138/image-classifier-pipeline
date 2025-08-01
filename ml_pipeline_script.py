import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import os
from PIL import Image
import requests
from io import BytesIO

# --- 1. Data Loading and Preprocessing ---

def load_and_preprocess_data(dataset_path, img_size=(64, 64)):
    """
    Loads and preprocesses image data from a specified path.

    This function is designed to be flexible and can be adapted to various
    image datasets. It handles loading images, resizing them to a consistent

    size, and converting them to a format suitable for model training.
    """
    images = []
    labels = []

    # Example: Loading data from a directory structure
    # This can be replaced with custom data loading logic
    for label, category in enumerate(os.listdir(dataset_path)):
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            for img_file in os.listdir(category_path):
                img_path = os.path.join(category_path, img_file)
                try:
                    # Using PIL for image loading and manipulation
                    img = Image.open(img_path).resize(img_size).convert('RGB')
                    images.append(np.array(img))
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading image {img_file}: {e}")

    # Convert to NumPy arrays and normalize pixel values
    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.array(labels, dtype=np.int64)

    return images, labels

# --- 2. Model Definition ---

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) for binary classification.

    This model is designed to be lightweight and fast, making it suitable
    for quick training and deployment. It consists of a few convolutional
    layers followed by fully connected layers.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 3. Training Function ---

def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    """
    Trains the model using the provided data loader and training parameters.

    This function iterates through the dataset for a specified number of epochs,
    performing forward and backward passes to train the model. It prints the
    loss at each epoch to monitor training progress.
    """
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")
    print("Finished Training")

# --- 4. Prediction Function ---

def predict(model, image_path):
    """
    Makes a prediction on a single image.

    This function loads an image, preprocesses it, and feeds it to the
    trained model to get a prediction. It returns the predicted class.
    """
    img = Image.open(image_path).resize((64, 64)).convert('RGB')
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs.data, 1)

    return predicted.item()

# --- Main Execution ---

if __name__ == "__main__":
    # Path to the dataset (replace with your dataset path)
    dataset_path = '/content/cats_vs_dogs'

    # Load and preprocess data
    images, labels = load_and_preprocess_data(dataset_path)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.from_numpy(X_train).permute(0, 3, 1, 2)
    y_train = torch.from_numpy(y_train)

    # Create DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model, criterion, and optimizer
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, criterion, optimizer)

    # Save the trained model
    torch.save(model.state_dict(), 'model.pth')