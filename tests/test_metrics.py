import numpy as np
import pandas as pd

from src import config
from src.data import label_thresholds, split
from src.evaluate import mae_by_range, metrics
from src.quality_metrics import band, image_diagnostics


def test_perfect_prediction_scores_perfectly():
    actual = np.array([10.0, 40.0, 55.0, 70.0, 90.0])
    result = metrics(actual, actual)
    assert result["mae"] == 0.0 and result["mse"] == 0.0
    assert np.isclose(result["srocc"], 1.0) and np.isclose(result["plcc"], 1.0)


def test_constant_offset_keeps_correlation_but_not_error():
    actual = np.array([10.0, 40.0, 55.0, 70.0, 90.0])
    result = metrics(actual + 5.0, actual)
    assert np.isclose(result["mae"], 5.0)
    assert np.isclose(result["plcc"], 1.0)


def test_band_edges():
    thresholds = {"low_below": 10.0, "high_above": 20.0}
    assert band(9.9, thresholds) == "Low"
    assert band(10.0, thresholds) == "Normal"   # not below the cut
    assert band(20.0, thresholds) == "Normal"   # not above the cut
    assert band(20.1, thresholds) == "High"


def test_mae_by_range_covers_every_image_once():
    actual = np.arange(0.0, 100.0)
    predicted = actual + 1.0
    groups = mae_by_range(predicted, actual, 33.0, 67.0)
    assert sum(g["images"] for g in groups.values()) == len(actual)
    assert all(np.isclose(g["mae"], 1.0) for g in groups.values())


def test_split_is_eighty_ten_ten_and_disjoint():
    frame = pd.DataFrame({"image_name": [f"{i}.jpg" for i in range(10073)],
                          "mos": np.linspace(4, 88, 10073)})
    train, val, test = split(frame)
    assert (len(train), len(val), len(test)) == (8058, 1007, 1008)
    names = [set(f["image_name"]) for f in (train, val, test)]
    assert not (names[0] & names[1]) and not (names[0] & names[2]) and not (names[1] & names[2])


def test_label_thresholds_sit_inside_the_scale_and_are_ordered():
    frame = pd.DataFrame({"mos": np.linspace(4, 88, 1000)})
    thresholds = label_thresholds(frame, config.MOS_MIN, config.MOS_MAX)
    assert config.MOS_MIN < thresholds["poor_below"] < thresholds["good_above"] < config.MOS_MAX


def test_diagnostics_separate_a_blurred_image_from_a_sharp_one():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, (384, 512, 3), dtype=np.uint8)
    flat = np.full((384, 512, 3), 128, dtype=np.uint8)
    assert image_diagnostics(noise)["sharpness"] > image_diagnostics(flat)["sharpness"]
    assert image_diagnostics(flat)["contrast"] < 1.0
