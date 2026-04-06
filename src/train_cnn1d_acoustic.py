import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------
# Acoustic Dataset
# -----------------------------
class AcousticDataset(Dataset):
    def __init__(self, csv_path, feature_dir):
        self.df = pd.read_csv(csv_path)
        self.feature_dir = feature_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        wav_path = self.df.iloc[idx]["path"]
        label = int(self.df.iloc[idx]["label"])

        # Convert wav filename to npz filename
        fname = os.path.basename(wav_path).replace(".wav", ".npz")
        feat_path = os.path.join(self.feature_dir, fname)

        data = np.load(feat_path)
        features = data["features"] if "features" in data.files else data["x"]

        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return features, label
# -----------------------------
# 1D-CNN Model Definition
# -----------------------------
class CNN1DAcoustic(nn.Module):
    def __init__(self, num_classes=8):
        super(CNN1DAcoustic, self).__init__()

        # 13*3 MFCC+deltas + ZCR + RMS + 12 chroma + 7 spectral_contrast = 60
        self.conv1 = nn.Conv1d(in_channels=60, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x shape: (B, T, F)
        x = x.permute(0, 2, 1)  # (B, F, T)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = self.gap(x)
        x = x.squeeze(-1)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)

        return x
# -----------------------------
# Load Datasets
# -----------------------------
train_dataset = AcousticDataset(
    csv_path="splits/train.csv",
    feature_dir="features/acoustic"
)

val_dataset = AcousticDataset(
    csv_path="splits/val.csv",
    feature_dir="features/acoustic"
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# -----------------------------
# Model Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN1DAcoustic(num_classes=8).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------
# Training Function
# -----------------------------
def train_one_epoch():
    model.train()
    running_loss = 0.0

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

# -----------------------------
# Validation Function
# -----------------------------
def evaluate():
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return acc, all_preds, all_labels

# -----------------------------
# Training Loop
# -----------------------------
os.makedirs("models", exist_ok=True)

best_acc = 0.0
epochs = 30

for epoch in range(epochs):
    train_loss = train_one_epoch()
    val_acc, preds, labels = evaluate()

    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "models/cnn1d_best.pt")
        print("✔ Best acoustic model saved")

# -----------------------------
# Final Evaluation
# -----------------------------
print("\nFinal Classification Report (Acoustic Model):")
print(classification_report(labels, preds))

print("Confusion Matrix:")
print(confusion_matrix(labels, preds)) 
