import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1DEncoder(nn.Module):
    def __init__(self, input_size):
        super(Conv1DEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear((input_size // 2) * 32, 128)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return x

class MLPClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TimeSeriesModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TimeSeriesModel, self).__init__()
        self.encoder = Conv1DEncoder(input_size)
        self.classifier = MLPClassifier(128, num_classes)  # 128 corresponds to the output of Conv1DEncoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
