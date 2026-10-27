"""MobileNetV2 with a small regression head."""
import tensorflow as tf

from . import config


def build_model(dropout=config.DROPOUT):
    """Returns (model, backbone). The backbone starts frozen and is called with
    training=False so its batch-normalisation layers stay in inference mode even
    after unfreezing."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=config.IMAGE_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=config.IMAGE_SIZE + (3,))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="linear")(x)
    return tf.keras.Model(inputs, outputs), base


def unfreeze(base, fine_tune_at=config.FINE_TUNE_AT):
    """Open the backbone from `fine_tune_at` onward. The caller must recompile."""
    base.trainable = True
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False
    return base


def compile_model(model, learning_rate):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="mse", metrics=["mae"])
    return model
