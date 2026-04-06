import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms
from PIL import Image

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =====================================================
# Dataset (Acoustic + Spectrogram)
# =====================================================
class FusionDataset(Dataset):
    def __init__(self, csv_path, acoustic_dir, spec_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.acoustic_dir = acoustic_dir
        self.spec_dir = spec_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row["label"])

        base = os.path.basename(row["path"]).replace(".wav", "")

        # Acoustic features
        acoustic_path = os.path.join(self.acoustic_dir, base + ".npz")
        z = np.load(acoustic_path)
        acoustic = z["features"] if "features" in z.files else z["x"]
        acoustic = torch.tensor(acoustic, dtype=torch.float32)

        # Spectrogram image
        spec_path = os.path.join(self.spec_dir, base + ".png")
        image = Image.open(spec_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return acoustic, image, label


# =====================================================
# Acoustic Branch (CNN + Attention)
# =====================================================
class AcousticBranch(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(60, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.attention = nn.Linear(128, 1)
        self.fc = nn.Linear(128, 128)

    def forward(self, x):
        x = x.permute(0, 2, 1)     # (B, F, T)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)     # (B, T, F)

        lstm_out, _ = self.lstm(x)

        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        return self.fc(context)


# =====================================================
# Spectral Branch (VGG16)
# =====================================================
class SpectralBranch(nn.Module):
    def __init__(self):
        super().__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        for param in vgg.features[:-15]:
            param.requires_grad = False

        self.features = vgg.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 128)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# =====================================================
# Fusion Model
# =====================================================
class FusionModel(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.acoustic = AcousticBranch()
        self.spectral = SpectralBranch()

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, acoustic_x, spectral_x):
        a = self.acoustic(acoustic_x)
        s = self.spectral(spectral_x)

        x = torch.cat([a, s], dim=1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# =====================================================
# Training Setup
# =====================================================
_use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if _use_cuda else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("Loading fusion dataset (paths from CSV; first batch will load .npz + PNG from disk)...", flush=True)
train_dataset = FusionDataset(
    "splits/train.csv",
    "features/acoustic",
    "features/specs",
    transform
)

val_dataset = FusionDataset(
    "splits/val.csv",
    "features/acoustic",
    "features/specs",
    transform
)
print(f"  Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}", flush=True)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    pin_memory=_use_cuda,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    pin_memory=_use_cuda,
)

print(f"Device: {device}", flush=True)
print(
    "Building FusionModel (loads VGG16 for spectral branch — first run may download weights; can take 1–2 min)...",
    flush=True,
)
sys.stdout.flush()
model = FusionModel().to(device)
print("Model ready. Training prints loss per batch; each epoch can be slow on CPU.\n", flush=True)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)


# =====================================================
# Training Loop
# =====================================================
os.makedirs("models", exist_ok=True)

best_acc = 0.0
epochs = 30

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    n_batches = len(train_loader)

    pbar = tqdm(train_loader, desc=f"Train {epoch + 1}/{epochs}", leave=True)
    for acoustic, image, labels in pbar:
        acoustic = acoustic.to(device, non_blocking=_use_cuda)
        image = image.to(device, non_blocking=_use_cuda)
        labels = labels.to(device, non_blocking=_use_cuda)

        optimizer.zero_grad()
        outputs = model(acoustic, image)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = train_loss / max(n_batches, 1)

    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for acoustic, image, labels in tqdm(val_loader, desc=f"Val {epoch + 1}/{epochs}", leave=True):
            acoustic = acoustic.to(device, non_blocking=_use_cuda)
            image = image.to(device, non_blocking=_use_cuda)
            labels = labels.to(device, non_blocking=_use_cuda)

            outputs = model(acoustic, image)
            pred = torch.argmax(outputs, dim=1)

            preds.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    acc = accuracy_score(targets, preds)

    print(
        f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Val Acc: {acc*100:.2f}%",
        flush=True,
    )

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "models/fusion_best.pt")
        print("✔ Best Fusion Model Saved")


# =====================================================
# Final Evaluation
# =====================================================
print("\nFinal Classification Report:")
print(classification_report(targets, preds))

print("Confusion Matrix:")
print(confusion_matrix(targets, preds))
