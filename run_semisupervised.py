import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from model import TimeSeriesClassifier
from data_loader import DrivingDataLoader_SemiSupervised

file_path = './data/driving.csv'
window_size = 30
step_size = 1

input_dim = 51  # X_window의 feature 수
hidden_dim = 64
encoder_output_dim = hidden_dim * ((window_size + 1) // 2**2)  # Conv 레이어 수에 맞춰 조정
num_classes = 10

num_epochs = 20
batch_size = 32
learning_rate = 0.001

confidence_threshold = 0.9  # Confidence threshold for pseudo-labeling
pseudo_label_epochs = 5  # Pseudo-label을 학습하는 반복 횟수

data_loader = DrivingDataLoader_SemiSupervised(file_path, window_size, step_size)
data_loader.load_data()
train_labeled_windows, train_unlabeled_windows, val_windows, test_windows = data_loader.get_splits()

def create_dataloader(windows, batch_size, shuffle=True, labeled=True):
    X, y = [], []
    if labeled:
        # Labeled 데이터인 경우 (X와 y를 모두 사용)
        for window_data, label in windows:
            X.append(window_data.values)
            y.append(label)
        X = torch.tensor(np.array(X), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.long)
        dataset = TensorDataset(X, y)
    else:
        # Unlabeled 데이터인 경우 (X만 사용)
        for window_data, _ in windows:
            X.append(window_data.values)
        X = torch.tensor(np.array(X), dtype=torch.float32)
        dataset = TensorDataset(X)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_labeled_loader = create_dataloader(train_labeled_windows, batch_size, labeled=True)
train_unlabeled_loader = create_dataloader(train_unlabeled_windows, batch_size, labeled=False)
val_loader = create_dataloader(val_windows, batch_size, shuffle=False, labeled=True)
test_loader = create_dataloader(test_windows, batch_size, shuffle=False, labeled=True)

model = TimeSeriesClassifier(input_dim=input_dim, hidden_dim=hidden_dim, encoder_output_dim=encoder_output_dim, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    
    # 1. Labeled 데이터 학습
    if train_labeled_loader is not None:
        for X_batch, y_batch in train_labeled_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # 2. Unlabeled 데이터에서 pseudo-labeling 학습
    if train_unlabeled_loader is not None:
        pseudo_labels = []
        pseudo_data = []
        
        # Unlabeled 데이터에 대해 confidence 기반으로 pseudo-label 생성
        with torch.no_grad():
            for X_batch in train_unlabeled_loader:
                X_batch = torch.stack(X_batch)
                X_batch = X_batch.squeeze()
                outputs = model(X_batch)
                probabilities = F.softmax(outputs, dim=1)  # 각 클래스에 대한 확률 계산
                max_probs, pseudo_classes = torch.max(probabilities, dim=1)  # 가장 높은 확률과 해당 클래스 추출

                # Confidence threshold 이상인 샘플에 대해 pseudo-label 저장
                high_confidence_mask = max_probs >= confidence_threshold
                if high_confidence_mask.any():
                    pseudo_labels.append(pseudo_classes[high_confidence_mask])
                    pseudo_data.append(X_batch[high_confidence_mask])

        # Pseudo-labeled 데이터를 새로운 labeled 데이터 로더로 학습
        if pseudo_labels and pseudo_data:
            pseudo_labels = torch.cat(pseudo_labels, dim=0)
            pseudo_data = torch.cat(pseudo_data, dim=0)
            pseudo_dataset = TensorDataset(pseudo_data, pseudo_labels)
            pseudo_loader = DataLoader(pseudo_dataset, batch_size=batch_size, shuffle=True)

            # Pseudo-labeled 데이터 학습 (Optional: 일부 에포크 동안 반복 학습 가능)
            for _ in range(pseudo_label_epochs):
                for X_batch, y_batch in pseudo_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
    
    # 3. Validation 성능 확인
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    if val_loader is not None:
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                # Accuracy 계산
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        val_accuracy = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss / len(val_loader)}, Validation Accuracy: {val_accuracy:.4f}")

# Test 성능 확인
test_loss = 0
correct = 0
total = 0
model.eval()
if test_loader is not None:
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()
            
            # Accuracy 계산
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    test_accuracy = correct / total
    print(f"Test Loss: {test_loss / len(test_loader)}, Test Accuracy: {test_accuracy:.4f}")
else:
    print("No test data available.")