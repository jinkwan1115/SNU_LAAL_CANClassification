
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import load_data
from model import TimeSeriesModel

# Load data
train_loader, test_loader = load_data("driving.csv")

# Model parameters
input_size = next(iter(train_loader))[0].shape[1]  # Dynamically get the input size
num_classes = len(set(next(iter(train_loader))[1].numpy()))  # Number of unique classes in the dataset

# Initialize model, loss function, and optimizer
model = TimeSeriesModel(input_size=input_size, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
