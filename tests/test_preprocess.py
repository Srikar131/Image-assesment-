import numpy as np

from src import config
from src.preprocess import denormalize_mos, normalize_mos, prepare_array


def test_normalise_round_trip():
    values = [0.0, 3.91, 58.73, 88.39, 100.0]
    assert np.allclose(denormalize_mos(normalize_mos(values)), values, atol=1e-3)


def test_normalise_maps_scale_ends_to_zero_and_one():
    assert np.isclose(normalize_mos([config.MOS_MIN])[0], 0.0)
    assert np.isclose(normalize_mos([config.MOS_MAX])[0], 1.0)


def test_prepare_array_gives_one_scaled_batch():
    batch = prepare_array(np.random.randint(0, 256, (384, 512, 3), dtype=np.uint8))
    assert batch.shape == (1, *config.IMAGE_SIZE, 3)
    # mobilenet_v2.preprocess_input scales to [-1, 1]
    assert -1.0001 <= float(np.min(batch)) and float(np.max(batch)) <= 1.0001


def test_prepare_array_drops_an_alpha_channel():
    assert prepare_array(np.zeros((100, 100, 4), dtype=np.uint8)).shape[-1] == 3
