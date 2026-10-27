"""Metrics and the plots that go into the reports."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from . import config
from .preprocess import build_dataset, denormalize_mos


def metrics(predicted, actual):
    """MAE and MSE on the MOS scale, plus rank (SROCC) and linear (PLCC)
    correlation. PLCC is plain Pearson, with no logistic mapping."""
    predicted = np.asarray(predicted, dtype="float64")
    actual = np.asarray(actual, dtype="float64")
    return {
        "images": int(len(actual)),
        "mae": float(np.mean(np.abs(predicted - actual))),
        "mse": float(np.mean((predicted - actual) ** 2)),
        "srocc": float(spearmanr(predicted, actual)[0]),
        "plcc": float(pearsonr(predicted, actual)[0]),
    }


def evaluate_split(model, frame, name, mos_min=config.MOS_MIN, mos_max=config.MOS_MAX,
                   result_dir=config.RESULT_DIR):
    ds = build_dataset(frame, training=False, mos_min=mos_min, mos_max=mos_max)
    predicted = denormalize_mos(model.predict(ds, verbose=1).squeeze(), mos_min, mos_max)
    actual = frame["mos"].values.astype("float32")

    results = metrics(predicted, actual)
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / f"{name}_metrics.json").write_text(json.dumps(results, indent=2))
    print(name, results)
    return predicted, actual, results


def mae_by_range(predicted, actual, low_cut, high_cut):
    """MAE split into the low, middle and high thirds of the score scale, which is
    where the error is least even."""
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    groups = {
        "low": actual <= low_cut,
        "middle": (actual > low_cut) & (actual < high_cut),
        "high": actual >= high_cut,
    }
    return {
        name: {"images": int(mask.sum()),
               "mae": float(np.mean(np.abs(predicted[mask] - actual[mask])))}
        for name, mask in groups.items()
    }


def plot_history(history, title, filename, figure_dir=config.FIGURE_DIR):
    hist = history.history if hasattr(history, "history") else history
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(hist["loss"], label="train")
    axes[0].plot(hist["val_loss"], label="validation")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("MSE loss"); axes[0].legend()
    axes[1].plot(hist["mae"], label="train")
    axes[1].plot(hist["val_mae"], label="validation")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("MAE (normalised scale)"); axes[1].legend()
    fig.suptitle(title)
    plt.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_dir / filename, dpi=200)
    plt.close(fig)


def plot_scatter(predicted, actual, name, mos_min=config.MOS_MIN, mos_max=config.MOS_MAX,
                 figure_dir=config.FIGURE_DIR):
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(actual, predicted, s=6, alpha=0.3, color="#4a7ab5")
    plt.plot([mos_min, mos_max], [mos_min, mos_max], color="#a2581c", linewidth=1)
    plt.xlabel("human MOS")
    plt.ylabel("predicted MOS")
    plt.title(f"{name}: predicted against human scores")
    plt.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_dir / f"{name}_scatter.png", dpi=200)
    plt.close()
