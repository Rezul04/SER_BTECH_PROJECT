import numpy as np, librosa, PIL.Image as Image, pandas as pd, pathlib, tqdm
SR = 16000; N_MELS = 128; IMG_SIZE = (224,224)
out_dir = pathlib.Path("features/specs"); out_dir.mkdir(parents=True, exist_ok=True)
meta = pd.read_csv("splits/metadata.csv")
for path in tqdm.tqdm(meta["path"].tolist()):
    y, sr = librosa.load(path, sr=SR)
    y = librosa.util.normalize(y)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=sr//2)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_min, S_max = S_db.min(), S_db.max()
    S_img = (255*(S_db - S_min)/(S_max - S_min + 1e-8)).astype(np.uint8)
    im = Image.fromarray(S_img).convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    fname = pathlib.Path(path).with_suffix(".png").name
    im.save(out_dir / fname)
print("Spectrograms saved to features/specs/")
