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
patience = 5  # Early stopping 기준 epoch 수

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

best_val_loss = float('inf')
epochs_no_improve = 0
best_model_path = './checkpoints/best_model.pth'
log_path = './logs/log.txt'

with open(log_path, 'w') as log_file:
    log_file.write("Epoch, Train Loss, Train Accuracy, Val Loss, Val Accuracy\n")

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
    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

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
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # 로그 파일에 기록
    with open(log_path, 'a') as log_file:
        log_file.write(f"{epoch + 1}, {avg_train_loss:.4f}, {train_accuracy:.2f}, {avg_val_loss:.4f}, {val_accuracy:.2f}\n")

    # Early Stopping 조건 확인
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path)  # 가장 좋은 모델 저장
        print("Model saved with Validation Loss: {:.4f}".format(best_val_loss))
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print("Early stopping triggered.")
        break

# Test 단계
model.load_state_dict(torch.load(best_model_path))  # 가장 좋은 모델 로드
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