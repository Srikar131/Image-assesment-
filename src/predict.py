"""Single-image inference: the path the Streamlit app takes."""
import json
import functools

import numpy as np
import tensorflow as tf

from . import config
from .preprocess import prepare_array, denormalize_mos
from .quality_metrics import band, image_diagnostics


@functools.lru_cache(maxsize=1)
def load_assets(model_path=config.MODEL_PATH,
                label_path=config.LABEL_THRESHOLDS_PATH,
                diagnostic_path=config.DIAGNOSTIC_THRESHOLDS_PATH):
    """Load the trained model and the two threshold files once per process."""
    for path in (model_path, label_path, diagnostic_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Unzip the Colab artifacts (iqa_artifacts.zip) "
                f"into the project folder first."
            )
    model = tf.keras.models.load_model(model_path)
    return model, json.loads(label_path.read_text()), json.loads(diagnostic_path.read_text())


def quality_level(mos, label_thresholds):
    if mos < label_thresholds["poor_below"]:
        return "Poor"
    if mos > label_thresholds["good_above"]:
        return "Good"
    return "Average"


def assess(image, assets=None):
    """`image` is a file path or an RGB array. Returns the predicted MOS, the same
    score out of 100, the quality level and the three diagnostics with their bands."""
    model, labels, diagnostics = assets or load_assets()

    if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
        array = np.asarray(tf.image.decode_jpeg(tf.io.read_file(str(image)), channels=3))
    else:
        array = np.asarray(image)

    prediction = model.predict(prepare_array(array), verbose=0)[0, 0]
    mos = float(denormalize_mos(prediction, labels["scale_min"], labels["scale_max"]))
    span = labels["scale_max"] - labels["scale_min"]
    measured = image_diagnostics(image)

    return {
        "mos": round(mos, 2),
        "score_100": round(100 * (mos - labels["scale_min"]) / span),
        "level": quality_level(mos, labels),
        "diagnostics": {
            name: {"value": round(value, 1), "band": band(value, diagnostics[name])}
            for name, value in measured.items()
        },
    }
