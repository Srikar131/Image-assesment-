"""Find the images, read the score file, split it and record the statistics."""
import json
import os

import numpy as np
import pandas as pd

from . import config


def find_image_dir(raw_dir=config.RAW_DIR):
    """The folder the KonIQ-10k jpegs were unzipped into."""
    first = next(iter(raw_dir.rglob("*.jpg")), None)
    if first is None:
        raise FileNotFoundError(f"no .jpg files under {raw_dir}")
    return first.parent


def load_scores(scores_csv, image_dir=None):
    """Read the score CSV, keep the rows whose image is actually on disk, and work
    out which scale the MOS column uses. Returns (frame, mos_min, mos_max)."""
    raw = pd.read_csv(scores_csv)
    image_col = [c for c in raw.columns if "image" in c.lower()][0]
    mos_col = [c for c in raw.columns if c.lower() == "mos"][0]

    image_dir = image_dir or find_image_dir()
    frame = raw[[image_col, mos_col]].rename(columns={image_col: "image_name", mos_col: "mos"})
    frame = frame.dropna()
    frame["path"] = frame["image_name"].map(lambda name: str(image_dir / name))

    found = frame["path"].map(os.path.exists)
    missing = int((~found).sum())
    if missing:
        print(f"{missing} rows dropped, image not on disk")
    frame = frame[found].reset_index(drop=True)

    # The distributions file uses 0-100; the 1-5 file would never exceed 5.5.
    mos_min, mos_max = (0.0, 100.0) if frame["mos"].max() > 5.5 else (1.0, 5.0)
    return frame, mos_min, mos_max


def dataset_stats(frame, mos_min, mos_max):
    return {
        "images_used": int(len(frame)),
        "scale_min": mos_min,
        "scale_max": mos_max,
        "mos_min": float(frame["mos"].min()),
        "mos_max": float(frame["mos"].max()),
        "mos_mean": float(frame["mos"].mean()),
        "mos_std": float(frame["mos"].std()),
        "mos_median": float(frame["mos"].median()),
    }


def split(frame, seed=config.SEED):
    """Random image-level 80/10/10 split. KonIQ-10k images are distinct, so no
    image can appear in two subsets."""
    shuffled = frame.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    return (
        shuffled.iloc[:n_train].reset_index(drop=True),
        shuffled.iloc[n_train:n_train + n_val].reset_index(drop=True),
        shuffled.iloc[n_train + n_val:].reset_index(drop=True),
    )


def save_splits(train, val, test, split_dir=config.SPLIT_DIR):
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [("train", train), ("val", val), ("test", test)]:
        frame[["image_name", "mos"]].to_csv(split_dir / f"{name}.csv", index=False)


def load_split(name, image_dir=None, split_dir=config.SPLIT_DIR):
    """Read back a saved split and re-attach the image paths."""
    frame = pd.read_csv(split_dir / f"{name}.csv")
    image_dir = image_dir or find_image_dir()
    frame["path"] = frame["image_name"].map(lambda n: str(image_dir / n))
    return frame


def split_stats(train, val, test):
    return {
        name: {
            "images": len(frame),
            "mos_mean": float(frame["mos"].mean()),
            "mos_std": float(frame["mos"].std()),
        }
        for name, frame in [("train", train), ("val", val), ("test", test)]
    }


def label_thresholds(train, mos_min, mos_max, tertiles=config.TERTILES):
    """Poor/Average/Good cuts, taken from the training MOS only."""
    low_cut, high_cut = np.percentile(train["mos"], list(tertiles))
    return {
        "scale_min": mos_min,
        "scale_max": mos_max,
        "poor_below": float(low_cut),
        "good_above": float(high_cut),
    }


def write_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return data
