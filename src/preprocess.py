"""Shared image preprocessing. Training and the Streamlit app both go through here,
so an image is treated the same way in both places."""
import numpy as np
import tensorflow as tf

from . import config

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input


def normalize_mos(values, mos_min=config.MOS_MIN, mos_max=config.MOS_MAX):
    """MOS on its own scale -> [0, 1], which is what the network is trained against."""
    return (np.asarray(values, dtype="float32") - mos_min) / (mos_max - mos_min)


def denormalize_mos(values, mos_min=config.MOS_MIN, mos_max=config.MOS_MAX):
    """Network output -> MOS on its own scale."""
    return np.asarray(values, dtype="float32") * (mos_max - mos_min) + mos_min


def read_image(path, training=False):
    """Decode, resize and scale one image inside a tf.data graph."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, config.IMAGE_SIZE)
    if training:
        image = tf.image.random_flip_left_right(image)
    return preprocess_input(image)


def prepare_array(image):
    """Same steps as read_image, for an RGB array the app already holds in memory.
    Returns a batch of one, ready for model.predict."""
    image = tf.convert_to_tensor(np.asarray(image)[..., :3], dtype=tf.float32)
    image = tf.image.resize(image, config.IMAGE_SIZE)
    return preprocess_input(image)[None, ...]


def build_dataset(frame, training=False, cache=False,
                  mos_min=config.MOS_MIN, mos_max=config.MOS_MAX):
    """tf.data pipeline over a frame with `path` and `mos` columns."""
    paths = frame["path"].values
    labels = normalize_mos(frame["mos"].values, mos_min, mos_max)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(frame), seed=config.SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (read_image(p, training), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(config.BATCH_SIZE)
    if cache:
        ds = ds.cache()
    return ds.prefetch(tf.data.AUTOTUNE)
