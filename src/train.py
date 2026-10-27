"""Two-phase training, run end to end. The Colab notebook does the same thing
cell by cell; this is the version that runs as a script."""
import json
import time

import tensorflow as tf

from . import config, data, evaluate
from .model import build_model, compile_model, unfreeze
from .preprocess import build_dataset
from .quality_metrics import compute_diagnostic_thresholds


def callbacks(checkpoint_name, log_name):
    return [
        tf.keras.callbacks.ModelCheckpoint(str(config.MODEL_DIR / checkpoint_name),
                                           monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                         patience=config.EARLY_STOPPING_PATIENCE,
                                         restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(str(config.LOG_DIR / log_name)),
    ]


def main(scores_csv=None, diagnostic_sample=1000):
    config.make_dirs()
    tf.keras.utils.set_random_seed(config.SEED)

    scores_csv = scores_csv or (config.RAW_DIR / "koniq10k_scores.csv")
    frame, mos_min, mos_max = data.load_scores(scores_csv)
    data.write_json(data.dataset_stats(frame, mos_min, mos_max),
                    config.RESULT_DIR / "dataset_stats.json")

    train_df, val_df, test_df = data.split(frame)
    data.save_splits(train_df, val_df, test_df)
    data.write_json(data.split_stats(train_df, val_df, test_df),
                    config.RESULT_DIR / "split_stats.json")

    train_ds = build_dataset(train_df, training=True, mos_min=mos_min, mos_max=mos_max)
    val_ds = build_dataset(val_df, cache=True, mos_min=mos_min, mos_max=mos_max)

    model, base = build_model()
    compile_model(model, config.LR_PHASE1)
    model.summary()

    start = time.time()
    history1 = model.fit(train_ds, validation_data=val_ds, epochs=config.EPOCHS_PHASE1,
                         callbacks=callbacks("phase1_best.keras", "phase1_log.csv"))
    minutes1 = (time.time() - start) / 60

    # Phase 2: open the top of the backbone at a much smaller learning rate.
    unfreeze(base)
    compile_model(model, config.LR_PHASE2)

    start = time.time()
    history2 = model.fit(train_ds, validation_data=val_ds, epochs=config.EPOCHS_PHASE2,
                         callbacks=callbacks("iqa_mobilenetv2.keras", "phase2_log.csv"))
    minutes2 = (time.time() - start) / 60
    model.save(config.MODEL_PATH)

    evaluate.plot_history(history1, "Phase 1: frozen backbone", "phase1_curves.png")
    evaluate.plot_history(history2, "Phase 2: fine-tuning", "phase2_curves.png")

    predicted, actual, _ = evaluate.evaluate_split(model, val_df, "val", mos_min, mos_max)
    evaluate.plot_scatter(predicted, actual, "validation", mos_min, mos_max)

    labels = data.write_json(data.label_thresholds(train_df, mos_min, mos_max),
                             config.LABEL_THRESHOLDS_PATH)
    data.write_json(evaluate.mae_by_range(predicted, actual,
                                          labels["poor_below"], labels["good_above"]),
                    config.RESULT_DIR / "val_mae_by_range.json")

    sample = train_df.sample(diagnostic_sample, random_state=config.SEED)
    data.write_json(compute_diagnostic_thresholds(sample["path"]),
                    config.DIAGNOSTIC_THRESHOLDS_PATH)

    data.write_json({
        "tensorflow": tf.__version__,
        "gpu": [d.name for d in tf.config.list_physical_devices("GPU")],
        "image_size": list(config.IMAGE_SIZE),
        "batch_size": config.BATCH_SIZE,
        "seed": config.SEED,
        "epochs_run_phase1": len(history1.history["loss"]),
        "epochs_run_phase2": len(history2.history["loss"]),
        "fine_tune_at": config.FINE_TUNE_AT,
        "minutes_phase1": round(minutes1, 1),
        "minutes_phase2": round(minutes2, 1),
    }, config.RESULT_DIR / "training_info.json")

    print("done. test set left untouched; run evaluate_split on it once, for the final report.")
    return model


if __name__ == "__main__":
    main()
