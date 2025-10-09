# Speech Emotion Recognition (SER) — Starter Repo


This repository contains a ready-to-run starter pipeline for a multimodal Speech Emotion Recognition project
(acoustic + spectrogram late fusion). The scripts produce acoustic feature stores, mel-spectrogram images,
simple baselines, and a fusion model.

## Structure
- `data/` (place RAVDESS & TESS raw wavs here)
- `features/acoustic/` (auto-generated .npz feature files)
- `features/specs/` (auto-generated spectrogram PNGs)
- `splits/` (metadata + train/val/test CSVs)
- `models/` (saved PyTorch checkpoints)
- `src/` (scripts)

## Quick start
1. Create & activate a virtual environment with Python 3.10+
2. `pip install -r requirements.txt`
3. Place datasets in `data/RAVDESS/` and `data/TESS/` (see dataset links in the earlier chat for download).
4. Run preprocessing and feature extraction scripts in this order:
   - `python src/build_metadata.py`
   - `python src/make_spectrograms.py`
   - `python src/extract_acoustic.py`
   - `python src/make_splits.py`
5. Train baselines or fusion model:
   - `python src/train_vgg_spec.py`
   - `python src/train_cnn1d_acoustic.py`
   - `python src/train_fusion.py`

## Notes
- The scripts are written for clarity over micro-optimizations; adapt paths/hyperparams as needed.
- Use GPU for reasonable training speed (check `torch.cuda.is_available()`).

## Contact
Ask me to tweak hyperparameters, add augmentation, or produce a Google-Colab notebook.
