import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
import umap

from model import TimeSeriesClassifier
from data_loader import DrivingDataLoader_Supervised

file_path = './data/driving.csv'
window_size = 50
step_size = 1
num_classes = 10
batch_size = 32

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

test_loader = create_dataloader(test_windows, batch_size, shuffle=False)

input_dim = 51  # X_window의 feature 수
hidden_dim = 64
encoder_output_dim = hidden_dim * ((window_size + 1) // 2**2)  # Conv 레이어 수에 맞춰 조정

best_model_path = './checkpoints/best_model.pth'

# 저장된 모델 가중치 로드
model = TimeSeriesClassifier(input_dim=input_dim, hidden_dim=hidden_dim, encoder_output_dim=encoder_output_dim, num_classes=num_classes)
model.load_state_dict(torch.load(best_model_path))
model.eval()

# Encoder만 추출
encoder = model.encoder  # 모델에서 encoder 부분을 분리
encoder.eval()

# Test set에서 representation 추출
representations = []
labels_list = []
with torch.no_grad():
    for inputs, labels in test_loader:
        encoded_outputs = encoder(inputs)
        representations.append(encoded_outputs.cpu().numpy())
        labels_list.append(labels.cpu().numpy())

# 리스트를 배열로 변환
representations = np.concatenate(representations, axis=0)
labels_list = np.concatenate(labels_list, axis=0)

# Argument parser
parser = argparse.ArgumentParser(description="2D Visualization with t-SNE or UMAP")
parser.add_argument("--method", type=str, default="tsne", choices=["tsne", "umap"], help="Dimensionality reduction method (tsne or umap)")
parser.add_argument("--perplexity", type=float, default=30.0, help="Perplexity for t-SNE")
parser.add_argument("--learning_rate", type=float, default=200.0, help="Learning rate for t-SNE")
parser.add_argument("--n_iter", type=int, default=1000, help="Number of iterations for t-SNE")
parser.add_argument("--n_neighbors", type=int, default=15, help="Number of neighbors for UMAP")
parser.add_argument("--min_dist", type=float, default=0.1, help="Minimum distance for UMAP")
args = parser.parse_args()

# Dimensionality reduction
if args.method == "tsne":
    reducer = TSNE(n_components=2, perplexity=args.perplexity, learning_rate=args.learning_rate, n_iter=args.n_iter, random_state=42)
    representations_2d = reducer.fit_transform(representations)
    method_name = "t-SNE"
elif args.method == "umap":
    reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist, random_state=42)
    representations_2d = reducer.fit_transform(representations)
    method_name = "UMAP"

# Color mapping and visualization
class_labels = [chr(65 + i) for i in range(10)]
colors = ListedColormap(["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
                         "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe"])

plt.figure(figsize=(10, 8))
scatter = plt.scatter(representations_2d[:, 0], representations_2d[:, 1], c=labels_list, cmap=colors, alpha=0.7)

unique_labels = np.unique(labels_list)
for label in unique_labels:
    plt.scatter([], [], color=colors(label), label=class_labels[label])

plt.legend(title="Class Labels (A-J)", loc="upper right")
plt.title(f"2D Representation of Test Set Encodings ({method_name})")
plt.xlabel(f"{method_name} Component 1")
plt.ylabel(f"{method_name} Component 2")

output_path = f"./visualization/{method_name.lower()}_visualization.png"
plt.savefig(output_path, format='png', dpi=300)
plt.show()

print(f"{method_name} visualization saved to {output_path}")