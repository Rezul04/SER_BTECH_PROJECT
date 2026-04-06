import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ==========================
# CONFIG
# ==========================
TRAIN_CSV = "splits/train_spec.csv"
VAL_CSV   = "splits/val_spec.csv"
IMG_SIZE = 64      # reduced for speed
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ==========================
# LOAD CSVs
# ==========================
print("Loading CSVs...")
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)

# ==========================
# IMAGE LOADER
# ==========================
def load_images(df):
    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row["filepath"].replace("\\", "/")
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.flatten() / 255.0
        X.append(img)
        y.append(row["label"])
    return np.array(X), np.array(y)

print("Loading images → features...")
X_train, y_train = load_images(train_df)
X_val, y_val     = load_images(val_df)

print("Shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val  :", X_val.shape,   "y_val  :", y_val.shape)

# ==========================
# MODELS
# ==========================
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=30),
    "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1),
    "SVM (Linear)":  SVC(kernel="linear", max_iter=3000)
}

# ==========================
# TRAIN & EVALUATE
# ==========================
results = {}

for name, model in models.items():
    print("\n" + "="*60)
    print(f"Training: {name}")
    print("="*60)

    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    acc = accuracy_score(y_val, preds)
    report = classification_report(y_val, preds, output_dict=True)

    results[name] = {
        "accuracy": acc,
        "f1_macro": report["macro avg"]["f1-score"]
    }

    # ---------- PRINT ----------
    print(f"Accuracy: {acc*100:.2f}%")
    print(classification_report(y_val, preds))

    # ---------- CONFUSION MATRIX ----------
    cm = confusion_matrix(y_val, preds)
    plt.figure(figsize=(7,6))
    plt.imshow(cm)
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.show()

# ==========================
# FINAL COMPARISON
# ==========================
print("\nFINAL MODEL COMPARISON")
print("="*60)
for k, v in results.items():
    print(f"{k:15s} | Acc: {v['accuracy']*100:.2f}% | F1: {v['f1_macro']:.3f}")
