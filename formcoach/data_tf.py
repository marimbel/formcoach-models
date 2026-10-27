"""tf.data input pipeline for Model A. Imported lazily so non-TF scripts don't need TensorFlow."""
from __future__ import annotations

from pathlib import Path

import tensorflow as tf

IMG = 224


def _load(path, training: bool):
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.cast(img, tf.float32)
    if training:
        # Random zoom 0.85-1.0 of the person crop, flip, photometric jitter.
        side = tf.shape(img)[0]
        s = tf.cast(tf.cast(side, tf.float32) * tf.random.uniform([], 0.85, 1.0), tf.int32)
        img = tf.image.random_crop(img, tf.stack([s, s, 3]))
        img = tf.image.resize(img, (IMG, IMG))
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 30.0)
        img = tf.image.random_contrast(img, 0.75, 1.25)
        img = tf.image.random_saturation(img, 0.7, 1.3)
        img = tf.clip_by_value(img, 0.0, 255.0)
    else:
        # Same as the app: whole square crop resized to 224.
        img = tf.image.resize(img, (IMG, IMG))
    return img  # 0..255 floats; MobileNetV3 has preprocessing built in


def make_dataset(df, root: Path, class_to_idx: dict, training: bool, batch: int = 32, seed: int = 663):
    paths = [str(Path(root) / p) for p in df.image_path]
    labels = [class_to_idx[c] for c in df.class_name]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (_load(p, training), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def load_images(paths) -> "tf.Tensor":
    return tf.stack([_load(tf.constant(str(p)), False) for p in paths])
