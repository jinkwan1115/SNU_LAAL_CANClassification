import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from model import TimeSeriesClassifier
from data_loader import DrivingDataLoader_Supervised

file_path = './data/driving.csv'
window_size = 50
step_size = 1

input_dim = 51  # X_window의 feature 수
hidden_dim = 64
encoder_output_dim = hidden_dim * ((window_size + 1) // 2**2)  # Conv 레이어 수에 맞춰 조정
num_classes = 10

num_epochs = 20
batch_size = 32
learning_rate = 0.001

data_loader = DrivingDataLoader_Supervised(file_path, window_size, step_size, test_size=0.2, val_size=0.1)
data_loader.load_data()
train_windows, val_windows, test_windows = data_loader.get_splits()

def create_dataloader(windows, batch_size, shuffle=True):
    X, y = [], []
    for window_data, label in windows:
        X.append(window_data.values)
        y.append(label)
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.long)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_loader = create_dataloader(train_windows, batch_size)
val_loader = create_dataloader(val_windows, batch_size, shuffle=False)
test_loader = create_dataloader(test_windows, batch_size, shuffle=False)

model = TimeSeriesClassifier(input_dim=input_dim, hidden_dim=hidden_dim, encoder_output_dim=encoder_output_dim, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_accuracy = 100. * correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

    # Validation 단계
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_accuracy = 100. * correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")

# Test 단계
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

test_accuracy = 100. * correct / total
print(f"Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%")