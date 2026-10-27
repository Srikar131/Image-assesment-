import numpy as np

from src import config
from src.model import build_model, unfreeze


def test_head_is_the_only_trainable_part_at_first():
    model, base = build_model()
    assert len(base.layers) == 154
    assert not base.trainable
    # Dense(128) 163,968 + Dense(1) 129
    assert sum(np.prod(w.shape) for w in model.trainable_weights) == 164_097


def test_output_is_one_number_per_image():
    model, _ = build_model()
    output = model(np.zeros((2, *config.IMAGE_SIZE, 3), dtype="float32"), training=False)
    assert output.shape == (2, 1)


def test_unfreeze_opens_only_the_top_of_the_backbone():
    model, base = build_model()
    unfreeze(base, fine_tune_at=config.FINE_TUNE_AT)
    assert all(not layer.trainable for layer in base.layers[:config.FINE_TUNE_AT])
    assert any(layer.trainable for layer in base.layers[config.FINE_TUNE_AT:])
