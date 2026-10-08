"""Train the crop-disease classifier.

Architecture (backbone selected by config `model.backbone`, currently
EfficientNetB0 for Raspberry Pi deployment):

    EfficientNet*(include_top=False) -> GlobalAveragePooling2D
                                      -> Dense(38, activation="softmax")

Shares the reference repository's architecture shape; only the backbone was
changed for Pi performance.

Preprocessing: see the PREPROCESSING CONTRACT comment in build_datasets().
Keras EfficientNet rescales internally, so no external 1/255 is applied.

Intended for Google Colab or any GPU machine. NOT for the Raspberry Pi.

Usage:
    python validate_dataset.py --save-mapping      # always run this first
    python train.py
    python train.py --batch-size 16 --epochs 30
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    build_label_mapping,
    class_names_to_index_list,
    load_config,
    load_production_labels,
    output_dir,
    output_path,
    save_json,
    scan_split,
    set_seed,
    strip_private_keys,
)

# TensorFlow is required here and only here, so validate_dataset.py stays
# importable on a lightweight machine.
try:
    import tensorflow as tf
except ImportError as e:  # pragma: no cover
    print("TensorFlow is required for training.\n"
          "  pip install -r requirements.txt\n"
          "Training is intended for Colab or a GPU machine, not the Raspberry Pi.",
          file=sys.stderr)
    raise SystemExit(2)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the crop-disease classifier.")
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--batch-size", type=int, default=None, help="override batch size")
    p.add_argument("--epochs", type=int, default=None, help="override max epochs")
    p.add_argument("--learning-rate", type=float, default=None, help="override LR")
    p.add_argument("--unfreeze", action="store_true",
                   help="fine-tune the backbone instead of freezing it")
    p.add_argument("--unfreeze-layers", type=int, default=0,
                   help="unfreeze the last N backbone layers")
    p.add_argument("--output", default=None, help="override output directory")
    return p.parse_args(argv)


# --------------------------------------------------------------------- model
def augment_batches(dataset, cfg: dict, height: int, width: int):
    """Augment on-device, per batch, inside the tf.data graph.

    Implemented as a single `tf.raw_ops.ImageProjectiveTransformV3`, which
    covers rotation, translation, shear and zoom in one fused op. That keeps
    the work on the accelerator and avoids materialising extra host-RAM copies.

    Images stay uint8 in the pipeline and are NOT divided by 255: EfficientNet's
    own stem handles scaling, so rescaling here would break the verified
    preprocessing contract.

    Applied to the training split only. Operations documented in config.yaml:
    rotation, width/height shift, shear, zoom, horizontal flip. No hue or
    saturation jitter, which would corrupt disease colour cues.
    """
    a = cfg["training"].get("augmentation", {}) or {}
    rotation = float(a.get("rotation_range", 0.0))
    wshift = float(a.get("width_shift_range", 0.0))
    hshift = float(a.get("height_shift_range", 0.0))
    shear = float(a.get("shear_range", 0.0))
    zoom = float(a.get("zoom_range", 0.0))
    flip = bool(a.get("horizontal_flip", False))

    if not any([rotation, wshift, hshift, shear, zoom, flip]):
        return dataset

    def _augment(images, labels):
        x = tf.cast(images, tf.float32)
        if flip:
            x = tf.image.random_flip_left_right(x)

        if any([rotation, wshift, hshift, shear, zoom]):
            # Sample the geometric parameters once per image, not per batch,
            # so every sample in a batch is transformed differently.
            shape = tf.shape(x)
            theta = (tf.random.uniform([shape[0]], -rotation, rotation)
                     * (math.pi / 180.0))
            phi = (tf.random.uniform([shape[0]], -shear, shear)
                   * (math.pi / 180.0))
            tx = tf.random.uniform([shape[0]], -wshift, wshift) * tf.cast(width, tf.float32)
            ty = tf.random.uniform([shape[0]], -hshift, hshift) * tf.cast(height, tf.float32)
            # zoom < 1 zooms in; sample a scale in [1-zoom, 1].
            scale = 1.0 - tf.random.uniform([shape[0]], 0.0, zoom)

            # Build a real 2x3 affine matrix: rot_m @ shear_m, where
            #   rot_m   = rotation by theta
            #   shear_m = shear by phi
            # Angles must be converted to cos/sin; putting the raw angle in the
            # matrix would make the linear part ~identity and the "rotation"
            # a silent no-op.
            #
            # These are named *_m, not `rotation`/`shear`: rebinding those names
            # to tensors would make them function-locals for all of _augment,
            # so the range checks above would raise UnboundLocalError.
            cos_t, sin_t = tf.math.cos(theta), tf.math.sin(theta)
            tan_p = tf.math.tan(phi)
            ones = tf.ones_like(theta)
            zeros = tf.zeros_like(theta)
            rot_m = tf.stack([
                tf.stack([cos_t, -sin_t, zeros], axis=-1),
                tf.stack([sin_t, cos_t, zeros], axis=-1),
            ], axis=1)                                   # (batch, 2, 3)
            shear_m = tf.stack([
                tf.stack([ones, tan_p, zeros], axis=-1),
                tf.stack([zeros, ones, zeros], axis=-1),
            ], axis=1)                                   # (batch, 2, 3)
            # Compose rot_m @ shear_m (2x3 each, homogeneous row implied).
            composed = tf.stack([
                tf.stack([rot_m[:, 0, 0] * shear_m[:, 0, 0]
                          + rot_m[:, 0, 1] * shear_m[:, 1, 0],
                          rot_m[:, 0, 0] * shear_m[:, 0, 1]
                          + rot_m[:, 0, 1] * shear_m[:, 1, 1],
                          tf.zeros_like(theta)], axis=-1),
                tf.stack([rot_m[:, 1, 0] * shear_m[:, 0, 0]
                          + rot_m[:, 1, 1] * shear_m[:, 1, 0],
                          rot_m[:, 1, 0] * shear_m[:, 0, 1]
                          + rot_m[:, 1, 1] * shear_m[:, 1, 1],
                          tf.zeros_like(theta)], axis=-1),
            ], axis=1)
            # `composed` maps input coordinates to output coordinates. The op
            # samples the source instead, so it wants the opposite direction:
            # a transform mapping output coordinates back to input ones. Fold
            # zoom into the forward matrix first (dividing by `scale` magnifies,
            # so once inverted the op receives scale < 1 and zooms IN, which is
            # what config.yaml documents), then invert the whole homogeneous
            # matrix. No rotation or shear is lost: only the direction flips.
            lin = composed[:, :, :2] / scale[:, None, None]
            fwd = tf.concat([
                tf.concat([lin, tf.stack([tx, ty], axis=-1)[:, :, None]],
                          axis=-1),
                tf.concat([tf.zeros_like(tx)[:, None], tf.zeros_like(ty)[:, None],
                           tf.ones_like(scale)[:, None]], axis=-1)[:, None, :],
            ], axis=1)                                   # (batch, 3, 3)
            inv = tf.linalg.inv(fwd)
            # Affine flat form [a0, a1, a2, b0, b1, b2, c0, c1]. The homogeneous
            # row is [0, 0, 1], so its first two entries are exactly c0, c1.
            transforms = tf.reshape(inv, [-1, 9])[:, :8]  # (batch, 8)

            x = tf.raw_ops.ImageProjectiveTransformV3(
                images=x,
                transforms=transforms,
                output_shape=shape[1:3],
                interpolation="NEAREST",
                fill_mode="NEAREST",
                fill_value=0.0,
            )
        x = tf.clip_by_value(x, 0.0, 255.0)
        return tf.cast(x, tf.uint8), labels

    # Order is non-deterministic on purpose: augmentation is random, and the
    # upstream shuffle already guarantees decorrelation across epochs.
    return dataset.map(_augment, num_parallel_calls=2, deterministic=False)


# Backbones selectable via config `model.backbone`. Kept explicit so an
# unsupported name fails loudly instead of silently using the wrong network.
BACKBONES = {
    "EfficientNetB0": "EfficientNetB0",
    "EfficientNetB1": "EfficientNetB1",
    "EfficientNetB2": "EfficientNetB2",
    "EfficientNetB3": "EfficientNetB3",
    "EfficientNetB4": "EfficientNetB4",
    "EfficientNetB5": "EfficientNetB5",
    "EfficientNetB6": "EfficientNetB6",
    "EfficientNetB7": "EfficientNetB7",
}


def build_model(cfg: dict, freeze: bool = True, unfreeze_layers: int = 0):
    """Backbone from config + GlobalAveragePooling2D + Dense(38, softmax)."""
    m = cfg["model"]
    backbone_name = m["backbone"]
    if backbone_name not in BACKBONES:
        raise SystemExit(
            f"FATAL: unknown backbone {backbone_name!r}. "
            f"Choose one of: {', '.join(sorted(BACKBONES))}"
        )
    applications = tf.keras.applications
    base = getattr(applications, BACKBONES[backbone_name])(
        include_top=m["include_top"],
        weights=m["weights"],
        input_shape=(m["input_height"], m["input_width"], 3),
    )

    if freeze and unfreeze_layers <= 0:
        # Reference behaviour: the backbone stays frozen and only the
        # 38-unit head learns. Much faster, and far less VRAM.
        base.trainable = False
    else:
        base.trainable = True
        if unfreeze_layers > 0:
            total = len(base.layers)
            for layer in base.layers[max(0, total - unfreeze_layers):]:
                layer.trainable = True
            print(f"Unfrozen last {unfreeze_layers} of {total} backbone layers")

    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(m["num_classes"],
                                    activation=m["activation"])(x)

    model = tf.keras.Model(inputs=base.input, outputs=outputs)
    return model


def build_datasets(cfg: dict, mapping: dict[str, int], labels: list[str]):
    """Load splits with class indices pinned to the production label order."""
    ds = cfg["dataset"]
    train_info = scan_split(cfg, "train")
    valid_info = scan_split(cfg, "valid")
    test_info = scan_split(cfg, "test")

    if train_info is None or valid_info is None:
        raise SystemExit("train/valid splits not found; run validate_dataset.py")

    h, w = cfg["model"]["input_height"], cfg["model"]["input_width"]

    # PREPROCESSING CONTRACT
    # ----------------------
    # Keras EfficientNet graphs contain their own `Rescaling(1/255)` stem plus
    # an ImageNet `Normalization`, and expect inputs in [0, 255]. Verified with
    # verify_graph.py; see README "Preprocessing contract".
    #
    # The production module (ai_disease_detection) divides by 255 and sends
    # [0, 1]. To keep that contract AND a correctly-fed backbone, the exported
    # graph is prefixed with `Rescaling(255)` by export_onnx.py, restoring
    # [0, 255] before EfficientNet's stem.
    #
    # So at TRAINING time we must feed [0, 255] (raw pixel values) to match
    # what the graph will see at INFERENCE time. Adding another external
    # 1/255 here would divide twice and desynchronise train from inference.
    # Hence: no external rescaling in the dataset pipeline.
    del ds  # dataset root already resolved during validation

    # The mapping pins output index -> production label index, so directory
    # sorting can never shift the class meaning:
    #   ordered_folders[i] is the dataset folder for labels.txt[i]
    #   Keras labels each class with its position in class_names
    #   => model output index i always means labels.txt[i]
    ordered_folders = [name for name, _ in sorted(mapping.items(),
                                                  key=lambda kv: kv[1])]
    expected_n = cfg["model"]["num_classes"]
    if len(ordered_folders) != expected_n:
        raise SystemExit(
            f"FATAL: mapping covers {len(ordered_folders)} classes, "
            f"expected {expected_n}. Run validate_dataset.py."
        )
    class_names = ordered_folders

    # Every split must expose exactly the same folders. Keras silently skips
    # folders absent from class_names, which would quietly shrink a split.
    for split_name, info in (("valid", valid_info), ("test", test_info)):
        if info is None:
            continue
        missing_folders = set(class_names) - set(info.class_dirs)
        extra_folders = set(info.class_dirs) - set(class_names)
        if missing_folders or extra_folders:
            raise SystemExit(
                f"FATAL: the {split_name} split does not match the train split.\n"
                f"  missing from {split_name}: {sorted(missing_folders)}\n"
                f"  unexpected in {split_name}: {sorted(extra_folders)}\n"
                "Fix the dataset before training; silently differing splits "
                "would corrupt the reported metrics."
            )

    # MEMORY SAFETY
    # -------------
    # The dataset is streamed from disk. Two things used to exhaust Colab RAM:
    #
    #   1. dataset.cache() with no argument caches the ENTIRE decoded dataset in
    #      RAM. At 224x224x3 float32 that is 70295 * 602112 bytes ~= 42 GB for
    #      train alone, plus ~10 GB for valid. That caused the session crash.
    #      There is no in-RAM cache anywhere now.
    #   2. prefetch(tf.data.AUTOTUNE) lets the runtime buffer arbitrarily many
    #      batches. A small fixed buffer keeps the pipeline fed without letting
    #      it grow unbounded.
    #
    # Augmentation runs per batch on-device (GPU when available), so it never
    # materialises an extra copy of the dataset. Images stay uint8 until the
    # model, and are NOT rescaled here: EfficientNet's stem does that.
    prefetch = int(cfg["training"].get("prefetch_batches", 2))
    buffer = int(cfg["training"].get("shuffle_buffer_batches", 200))
    batch_size = cfg["training"]["batch_size"]

    def make(info, shuffle):
        # batch_size=None: the shuffle below has to run on individual images.
        # Batching first would make buffer_size count batches, turning the
        # 4096-IMAGE cap into 4096 batches (~79 GB at 32 x 602112 bytes), and
        # filling it exhausts Colab host RAM. The cap is only correct in image
        # units, so load unbatched, shuffle, then batch.
        dataset = tf.keras.utils.image_dataset_from_directory(
            str(info.root),
            image_size=(h, w),
            batch_size=None,
            label_mode="categorical",
            class_names=class_names,   # index i -> labels.txt[i]
            shuffle=False,            # shuffling is done explicitly below
        )
        if shuffle:
            # Explicit reshuffle each epoch with a bounded shuffle buffer.
            # A huge buffer here would itself be a RAM sink, so the cap binds:
            # min(200 * 32, 4096) = 4096 images, never 4096 batches.
            dataset = dataset.shuffle(
                buffer_size=min(buffer * batch_size, 4096),
                reshuffle_each_iteration=True,
            )
        dataset = dataset.batch(batch_size=batch_size)
        if shuffle:
            dataset = augment_batches(dataset, cfg, h, w)
        # No .cache(): stream from disk every pass.
        return dataset.prefetch(prefetch)

    # Train shuffled, validation and test in a deterministic order so repeated
    # runs are comparable.
    train_ds = make(train_info, True)
    valid_ds = make(valid_info, False)

    test_ds = None
    if test_info is not None:
        test_ds = make(test_info, False)

    # No external Rescaling here on purpose: Keras EfficientNet already
    # contains Rescaling(1/255) + ImageNet Normalization and expects [0, 255].
    # See the PREPROCESSING CONTRACT note above.
    return train_ds, valid_ds, test_ds


def build_callbacks(cfg: dict, model, dest: Path):
    c = cfg["training"]["callbacks"]
    cbs = []

    if c.get("checkpoint", {}).get("enabled", True):
        cbs.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=str(dest),
            monitor=c.get("checkpoint", {}).get("monitor", "val_loss"),
            save_best_only=c.get("checkpoint", {}).get("save_best_only", True),
            save_weights_only=c.get("checkpoint", {}).get("save_weights_only", False),
            verbose=1,
        ))
    if c.get("early_stopping", {}).get("enabled", True):
        cbs.append(tf.keras.callbacks.EarlyStopping(
            monitor=c.get("early_stopping", {}).get("monitor", "val_loss"),
            patience=c.get("early_stopping", {}).get("patience", 3),
            restore_best_weights=c.get("early_stopping", {}).get(
                "restore_best_weights", True),
            verbose=1,
        ))
    if c.get("reduce_lr", {}).get("enabled", True):
        cbs.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor=c.get("reduce_lr", {}).get("monitor", "val_loss"),
            factor=c.get("reduce_lr", {}).get("factor", 0.5),
            patience=c.get("reduce_lr", {}).get("patience", 1),
            min_lr=c.get("reduce_lr", {}).get("min_lr", 1e-6),
            verbose=1,
        ))
    return cbs


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.learning_rate:
        cfg["training"]["learning_rate"] = args.learning_rate

    seed = cfg["training"]["seed"]
    set_seed(seed)
    print("=" * 68)
    print(f"TRAINING  seed={seed}  batch={cfg['training']['batch_size']}  "
          f"epochs<={cfg['training']['epochs']}")
    print("=" * 68)

    # --------------------------------------------------------- label mapping
    labels = load_production_labels(cfg)
    train_info = scan_split(cfg, "train")
    if train_info is None:
        print("FATAL: train split not found. Run validate_dataset.py.",
              file=sys.stderr)
        return 3
    # Also needed here for the batch-count report below; build_datasets()
    # resolves its own copy internally.
    valid_info = scan_split(cfg, "valid")
    if valid_info is None:
        print("FATAL: valid split not found. Run validate_dataset.py.",
              file=sys.stderr)
        return 3
    mapping, missing, unexpected = build_label_mapping(train_info.class_dirs, labels)
    if missing or unexpected:
        print("FATAL: class mapping is not safe. Run validate_dataset.py.",
              file=sys.stderr)
        return 3

    # Cross-check against the mapping validate_dataset.py wrote, so training
    # cannot silently use a different class ordering than the one reviewed.
    mapping_file = output_dir(cfg) / cfg["output"]["label_mapping"]
    saved = None
    if mapping_file.is_file():
        import json

        saved = json.loads(mapping_file.read_text(encoding="utf-8"))
        saved_mapping = saved.get("mapping", {})
        if saved_mapping and saved_mapping != mapping:
            disagreements = {
                k: (saved_mapping.get(k), mapping.get(k))
                for k in set(saved_mapping) | set(mapping)
                if saved_mapping.get(k) != mapping.get(k)
            }
            print("FATAL: the dataset no longer matches "
                  f"{mapping_file.name}.", file=sys.stderr)
            print(f"  differing classes (saved, current): {disagreements}",
                  file=sys.stderr)
            print("  Re-run validate_dataset.py --save-mapping, then train.",
                  file=sys.stderr)
            return 6
    print(f"Class mapping verified: {len(mapping)} classes, "
          f"indices 0-{max(mapping.values())} match labels.txt")

    # ------------------------------------------------------------- GPU check
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs detected: {len(gpus)}")
    for g in gpus:
        print(f"  {g.name}")
    if not gpus:
        print("WARNING: no GPU. EfficientNetB7 on CPU is impractically slow;")
        print("         reduce epochs or use Colab.")
    else:
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass

    # ----------------------------------------------------------------- data
    train_ds, valid_ds, test_ds = build_datasets(cfg, mapping, labels)
    if test_ds is None:
        print("NOTE: no test split found. Test metrics will require either a")
        print("      test directory, or an explicit --split valid in")
        print("      evaluate.py. Validation figures are NOT test figures.")
    # Count from the scan, not from the tf.data pipeline: materialising the
    # datasets with list() re-decodes and caches every image, which is slow and
    # memory-hungry for no informational gain.
    train_count = train_info.image_count
    valid_count = valid_info.image_count
    batch_size = cfg["training"]["batch_size"]
    print(f"train images: {train_count}  "
          f"train batches: {math.ceil(train_count / batch_size)}")
    print(f"valid images: {valid_count}  "
          f"valid batches: {math.ceil(valid_count / batch_size)}")

    # ---------------------------------------------------------------- model
    model = build_model(
        cfg,
        freeze=not args.unfreeze and cfg["model"]["freeze_backbone"],
        unfreeze_layers=args.unfreeze_layers,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=cfg["training"]["learning_rate"]),
        loss=cfg["training"]["loss"],
        metrics=cfg["training"]["metrics"],
    )
    model.summary()

    out_dir = output_dir(cfg)
    dest = out_dir / cfg["output"]["model_h5"]

    # Record the exact mapping this run trained on, so evaluate.py and any
    # later export can prove they share it.
    save_json({
        "num_classes": len(mapping),
        "exact_order_match": class_names_to_index_list(mapping) == labels,
        "mapping": mapping,
        "ordered_dataset_classes": class_names_to_index_list(mapping),
        "production_labels": labels,
    }, mapping_file)

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=cfg["training"]["epochs"],
        callbacks=build_callbacks(cfg, model, dest),
    )

    # ------------------------------------------------------------- artifacts
    # checkpoint already wrote the best model, but save explicitly so the file
    # exists even if checkpointing was disabled in config.
    if not dest.is_file():
        model.save(dest)
        print(f"Saved Keras model: {dest}")

    import yaml
    (out_dir / cfg["output"]["training_config"]).write_text(
        yaml.safe_dump(strip_private_keys(cfg), sort_keys=False), encoding="utf-8")
    (out_dir / cfg["output"]["class_names"]).write_text(
        "\n".join(labels) + "\n", encoding="utf-8")

    save_json({
        "epochs_run": len(history.history.get("loss", [])),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "best_val_loss": float(min(history.history["val_loss"])),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "batch_size": cfg["training"]["batch_size"],
        "learning_rate": cfg["training"]["learning_rate"],
        "seed": seed,
        "note": "Validation figures only. Run evaluate.py for test metrics.",
    }, out_dir / cfg["output"]["metrics"])

    print(f"\nArtifacts written to {out_dir}")
    print("Next: python evaluate.py   then   python export_onnx.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())