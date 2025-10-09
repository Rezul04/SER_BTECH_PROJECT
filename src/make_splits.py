import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
meta = pd.read_csv("splits/metadata.csv")
emotions = sorted(meta["emotion"].unique())
emap = {e:i for i,e in enumerate(emotions)}
meta["label"] = meta["emotion"].map(emap)
def split_df(df):
    unique_spk = df["speaker"].unique()
    if len(unique_spk) < 2:
        # Not enough speakers to split
        return df, df, df

    spk_train, spk_temp = train_test_split(unique_spk, test_size=0.3, random_state=42)
    spk_val, spk_test = train_test_split(spk_temp, test_size=0.5, random_state=42)

    return (
        df[df["speaker"].isin(spk_train)],
        df[df["speaker"].isin(spk_val)],
        df[df["speaker"].isin(spk_test)],
    )

r_tr, r_va, r_te = split_df(meta[meta["dataset"]=="RAVDESS"]) 
t_tr, t_va, t_te = split_df(meta[meta["dataset"]=="TESS"]) 
train = pd.concat([r_tr,t_tr]).sample(frac=1, random_state=42)
val   = pd.concat([r_va,t_va]).sample(frac=1, random_state=42)
test  = pd.concat([r_te,t_te]).sample(frac=1, random_state=42)
train.to_csv("splits/train.csv", index=False); val.to_csv("splits/val.csv", index=False); test.to_csv("splits/test.csv", index=False)
print("Saved train/val/test splits to splits/")
