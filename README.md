# Speech Emotion Recognition (SER) - Multimodal Pipeline

This repository implements a Speech Emotion Recognition workflow using:
- acoustic time-series features, and
- mel-spectrogram image features,

with both single-branch baselines and a late-fusion multimodal model.

Primary datasets:
- RAVDESS
- TESS

## Repository Layout

- `data/` - raw WAV files
  - `data/RAVDESS/`
  - `data/TESS/`
- `features/acoustic/` - extracted acoustic features (`.npz`)
- `features/specs/` - mel-spectrogram images (`.png`)
- `splits/` - metadata and train/val/test CSVs
- `models/` - saved model checkpoints
- `src/` - preprocessing and training scripts

## Environment Setup

Use Python 3.10+.

Windows PowerShell:

```powershell
cd d:\ser_repo\ser_repo
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data Preparation Pipeline (Run in Order)

After placing dataset WAV files in `data/RAVDESS/` and `data/TESS/`, run:

```powershell
python src/build_metadata.py
python src/make_spectrograms.py
python src/extract_acoustic.py
python src/make_splits.py
python src/make_spec_csv.py
```

What each script does:

1. `build_metadata.py`
   - Scans RAVDESS and TESS WAV paths.
   - Builds `splits/metadata.csv` with columns:
     `path, dataset, speaker, emotion`.

2. `make_spectrograms.py`
   - Loads audio at 16 kHz.
   - Creates mel-spectrograms and saves 224x224 RGB PNGs to `features/specs/`.

3. `extract_acoustic.py`
   - Extracts acoustic features per frame:
     MFCC + delta + delta2, ZCR, RMS, chroma, spectral contrast.
   - Pads/truncates to 200 frames.
   - Saves `.npz` files with key `features` in `features/acoustic/`.

4. `make_splits.py`
   - Creates train/val/test CSVs under `splits/` with numeric labels.

5. `make_spec_csv.py`
   - Creates `splits/train_spec.csv` and `splits/val_spec.csv`
     for image-based training scripts.

## Implemented Models

### 1) Acoustic 1D CNN baseline
- Script: `src/train_cnn1d_acoustic.py`
- Input: acoustic `.npz` features (`T x 60`)
- Output checkpoint: `models/cnn1d_best.pt`

Run:

```powershell
python src/train_cnn1d_acoustic.py
```

### 2) Spectrogram VGG16 baseline
- Script: `src/train_vgg_spec.py`
- Input: `train_spec.csv` / `val_spec.csv`
- Output checkpoint: `models/vgg_spec_best.pt`

Run:

```powershell
python src/train_vgg_spec.py
```

### 3) Multimodal late-fusion model (final hybrid)
- Script: `src/train_fusion.py`
- Acoustic branch: CNN -> BiLSTM -> attention
- Spectral branch: VGG16 feature encoder
- Fusion: concatenation -> dense classifier
- Output checkpoint: `models/fusion_best.pt`

Run:

```powershell
python src/train_fusion.py
```

### 4) BiLSTM-only acoustic baseline
- Script: `src/train_bilstm_acoustic.py`
- Input: acoustic `.npz` features only
- Output checkpoint: `models/bilstm_acoustic_best.pt`

Run:

```powershell
python src/train_bilstm_acoustic.py
```

### 5) Classical ML baselines on spectrogram pixels
- Script: `src/step4_ml_models.py`
- Models: Decision Tree, Random Forest, linear SVM
- Input: flattened 64x64 spectrogram images

Run:

```powershell
python src/step4_ml_models.py
```

If OpenCV is missing:

```powershell
pip install opencv-python
```

## Notes on Training Speed

- If running on CPU, VGG16 and fusion training can appear slow per epoch.
- Scripts include progress output, but first run may still pause while downloading pretrained VGG16 weights.
- GPU is strongly recommended.

## Common Troubleshooting

- No output for a long time:
  - wait for first epoch start, especially on CPU.
  - check whether model weights are being downloaded on first run.

- Feature key mismatch in `.npz`:
  - current extractor saves `features`.
  - training loaders also support legacy files containing key `x`.

- Input channel mismatch:
  - acoustic feature dimension is 60 channels for current extraction settings.

## Typical Workflow

1. Prepare data once.
2. Train baseline models.
3. Train fusion model.
4. Compare validation metrics and confusion matrices.
5. Add new experimental models under `src/` using the same splits and features.
