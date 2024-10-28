import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch

class DrivingDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_data(file_path, test_size=0.3, batch_size=32):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Drop unnecessary columns
    df = df.drop(['Time(s)', 'PathOrder'], axis=1)

    # Feature and target split
    X = df.drop('Class', axis=1)
    y = df['Class'].astype('category').cat.codes  # Encode class labels if categorical

    # Handle missing values by filling with former value
    X = X.fillna(method='bfill')

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    # Create Dataset objects
    train_dataset = DrivingDataset(X_train, y_train)
    test_dataset = DrivingDataset(X_test, y_test)

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
