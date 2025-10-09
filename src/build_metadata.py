import re, csv, pathlib, pandas as pd
ROOT = pathlib.Path("data")
OUT = pathlib.Path("splits/metadata.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)
EMO_MAP_RAV = {
    '01':'neutral','02':'calm','03':'happy','04':'sad',
    '05':'angry','06':'fearful','07':'disgust','08':'surprised'
}
RAV_PATTERN = re.compile(r"(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.wav")
TESS_MAP = {'angry':'angry', 'disgust':'disgust','fear':'fearful','happy':'happy','neutral':'neutral','ps':'surprised','sad':'sad'}
rows = []
# RAVDESS
for p in ROOT.glob("RAVDESS/**/*.wav"):
    m = RAV_PATTERN.search(p.name)
    if not m: continue
    emotion_id = m.group(3)
    emotion = EMO_MAP_RAV.get(emotion_id)
    actor = f"Actor_{m.group(7)}"
    rows.append([str(p), "RAVDESS", actor, emotion])
# TESS
for p in ROOT.glob("TESS/**/*.wav"):
    name = p.name.lower()
    emo = None
    for k,v in TESS_MAP.items():
        if k in name:
            emo = v; break
    if emo:
        speaker = p.parts[-2]
        rows.append([str(p), "TESS", speaker, emo])
with OUT.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["path","dataset","speaker","emotion"])
    w.writerows(rows)
print(f"Wrote {OUT} with {len(rows)} rows")
