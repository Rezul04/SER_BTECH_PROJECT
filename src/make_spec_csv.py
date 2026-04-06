import pandas as pd
import os

def make_spec_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    rows = []
    for _, row in df.iterrows():
        wav_path = row["path"]
        label = row["label"]

        # Extract filename only
        fname = os.path.basename(wav_path).replace(".wav", ".png")

        spec_path = os.path.join("features", "specs", fname)
        rows.append([spec_path, label])

    out_df = pd.DataFrame(rows, columns=["filepath", "label"])
    out_df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")

make_spec_csv("splits/train.csv", "splits/train_spec.csv")
make_spec_csv("splits/val.csv", "splits/val_spec.csv")
