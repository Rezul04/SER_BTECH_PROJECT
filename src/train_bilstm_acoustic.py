import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class AcousticDataset(Dataset):
    def __init__(self, csv_path, feature_dir):
        self.df = pd.read_csv(csv_path)
        self.feature_dir = feature_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        wav_path = self.df.iloc[idx]["path"]
        label = int(self.df.iloc[idx]["label"])
        fname = os.path.basename(wav_path).replace(".wav", ".npz")
        feat_path = os.path.join(self.feature_dir, fname)
        data = np.load(feat_path)
        features = data["features"] if "features" in data.files else data["x"]
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return features, label


class BiLSTMAcoustic(nn.Module):
    """Sequence classifier: raw acoustic frames -> BiLSTM -> mean pool -> MLP. No CNN."""

    def __init__(self, input_size=60, num_classes=8, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # bidirectional -> 2 * hidden_size
        self.fc1 = nn.Linear(2 * hidden_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)
        h = self.dropout(self.relu(self.fc1(pooled)))
        return self.fc2(h)


print("Loading acoustic dataset from splits and .npz features...", flush=True)
train_dataset = AcousticDataset("splits/train.csv", "features/acoustic")
val_dataset = AcousticDataset("splits/val.csv", "features/acoustic")
print(f"  Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}", flush=True)

_use_cuda = torch.cuda.is_available()
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=_use_cuda)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=_use_cuda)

device = torch.device("cuda" if _use_cuda else "cpu")
print(f"Device: {device}", flush=True)
sys.stdout.flush()
model = BiLSTMAcoustic().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train_one_epoch():
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc="Train", leave=False)
    for features, labels in pbar:
        features = features.to(device, non_blocking=_use_cuda)
        labels = labels.to(device, non_blocking=_use_cuda)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / len(train_loader)


def evaluate():
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, labels in tqdm(val_loader, desc="Val", leave=False):
            features = features.to(device, non_blocking=_use_cuda)
            labels = labels.to(device, non_blocking=_use_cuda)
            outputs = model(features)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds), all_preds, all_labels


os.makedirs("models", exist_ok=True)
best_acc = 0.0
epochs = 30

for epoch in range(epochs):
    train_loss = train_one_epoch()
    val_acc, preds, labels = evaluate()
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "models/bilstm_acoustic_best.pt")
        print("✔ Best BiLSTM model saved")

print("\nFinal Classification Report (BiLSTM acoustic only):")
print(classification_report(labels, preds))
print("Confusion Matrix:")
print(confusion_matrix(labels, preds))
