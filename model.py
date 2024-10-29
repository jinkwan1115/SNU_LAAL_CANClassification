import torch
import torch.nn as nn

class Conv1DEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, kernel_size=3):
        super(Conv1DEncoder, self).__init__()
        layers = []
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            layers.append(nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
        self.conv = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, seq_len, feature_dim) -> (batch, feature_dim, seq_len)
        x = self.conv(x)
        x = self.flatten(x)
        return x


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TimeSeriesClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoder_output_dim, num_classes, num_layers=2, kernel_size=3):
        super(TimeSeriesClassifier, self).__init__()
        self.encoder = Conv1DEncoder(input_dim, hidden_dim, num_layers, kernel_size)
        self.classifier = MLPClassifier(encoder_output_dim, hidden_dim, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x