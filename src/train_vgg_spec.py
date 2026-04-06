import os
import sys

import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


class SpectrogramDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["filepath"]
        label = int(self.df.iloc[idx]["label"])

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("Loading CSVs and image paths...", flush=True)
train_dataset = SpectrogramDataset("splits/train_spec.csv", transform)
val_dataset = SpectrogramDataset("splits/val_spec.csv", transform)
print(f"  Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}", flush=True)

_use_cuda = torch.cuda.is_available()
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

device = torch.device("cuda" if _use_cuda else "cpu")
print(f"Device: {device} (VGG on CPU is slow — first epoch may take several minutes.)", flush=True)

print("Loading VGG16 (ImageNet weights download on first run can take a while)...", flush=True)
sys.stdout.flush()
vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
print("Model weights ready.", flush=True)

for param in vgg.features[:-5]:
    param.requires_grad = False

vgg.classifier[6] = nn.Linear(4096, 8)
vgg = vgg.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg.parameters(), lr=3e-5)

def train_one_epoch(epoch_idx, n_epochs):
    vgg.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Train {epoch_idx}/{n_epochs}", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=_use_cuda), labels.to(device, non_blocking=_use_cuda)
        optimizer.zero_grad()
        outputs = vgg(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / len(train_loader)

def evaluate():
    vgg.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Val", leave=False):
            images, labels = images.to(device, non_blocking=_use_cuda), labels.to(device, non_blocking=_use_cuda)
            outputs = vgg(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return acc, all_preds, all_labels

os.makedirs("models", exist_ok=True)
best_acc = 0.0
epochs = 15

for epoch in range(epochs):
    print(f"\n--- Epoch {epoch + 1}/{epochs} ---", flush=True)
    train_loss = train_one_epoch(epoch + 1, epochs)
    val_acc, preds, labels = evaluate()
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(vgg.state_dict(), "models/vgg_spec_best.pt")
        print("✔ Best model saved")

# =============================
# FINAL EVALUATION (BEST MODEL)
# =============================
print("\nEvaluating best saved model...")

vgg.load_state_dict(torch.load("models/vgg_spec_best.pt", map_location=device))
vgg.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Final eval"):
        images = images.to(device, non_blocking=_use_cuda)
        labels = labels.to(device, non_blocking=_use_cuda)

        outputs = vgg(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nFinal Classification Report:")
print(classification_report(all_labels, all_preds))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

