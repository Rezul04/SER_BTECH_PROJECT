import numpy as np, librosa, pathlib, tqdm, pandas as pd
SR = 16000; HOP = 512; N_MFCC = 13; T_TARGET = 200
meta = pd.read_csv("splits/metadata.csv")
out_dir = pathlib.Path("features/acoustic"); out_dir.mkdir(parents=True, exist_ok=True)
def pad_trunc(feat, T=T_TARGET):
    T_in = feat.shape[1]
    if T_in == T: return feat
    if T_in > T: return feat[:, :T]
    pad = np.zeros((feat.shape[0], T - T_in), dtype=feat.dtype)
    return np.concatenate([feat, pad], axis=1)
for path in tqdm.tqdm(meta["path"].tolist()):
    y, sr = librosa.load(path, sr=SR)
    y = librosa.util.normalize(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP)
    d1 = librosa.feature.delta(mfcc); d2 = librosa.feature.delta(mfcc, order=2)
    mfcc_all = np.vstack([mfcc, d1, d2])
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP)
    rmse = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP)
    spec_con = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=HOP)
    feats = np.vstack([mfcc_all, zcr, rmse, chroma, spec_con])
    feats = pad_trunc(feats)
    feats = feats.T.astype(np.float32)
    rel = pathlib.Path(path).with_suffix(".npz").name
    np.savez_compressed(out_dir / rel, features=feats)
print("Acoustic features saved to features/acoustic/")
