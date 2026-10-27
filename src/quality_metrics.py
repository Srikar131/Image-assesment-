"""OpenCV measurements shown next to the predicted score. The model does not use them."""
import json

import cv2
import numpy as np

from . import config


def image_diagnostics(image):
    """Sharpness, brightness and contrast of one image, measured on a grayscale
    copy at a fixed size so the numbers are comparable between images.
    `image` is a path or an RGB array."""
    if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
        bgr = cv2.imread(str(image))
        if bgr is None:
            raise ValueError(f"could not read image: {image}")
    else:
        bgr = cv2.cvtColor(np.asarray(image)[..., :3].astype("uint8"), cv2.COLOR_RGB2BGR)

    bgr = cv2.resize(bgr, config.DIAGNOSTIC_SIZE)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return {
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
    }


def band(value, thresholds):
    """Low / Normal / High against one diagnostic's tertile cuts."""
    if value < thresholds["low_below"]:
        return "Low"
    if value > thresholds["high_above"]:
        return "High"
    return "Normal"


def compute_diagnostic_thresholds(paths, low=config.TERTILES[0], high=config.TERTILES[1]):
    """Tertile bands for each diagnostic, computed over training images only."""
    measured = [image_diagnostics(p) for p in paths]
    return {
        name: {
            "low_below": float(np.percentile([m[name] for m in measured], low)),
            "high_above": float(np.percentile([m[name] for m in measured], high)),
        }
        for name in ("sharpness", "brightness", "contrast")
    }


def load_thresholds(path=config.DIAGNOSTIC_THRESHOLDS_PATH):
    return json.loads(path.read_text())
