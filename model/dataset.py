import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.data import AUTOTUNE

from config import BATCH, DATASET_DIR, STEPS, input_width, input_height, class_mapping


class CarouselDataset:
    def __init__(self, annotation_path='./data/train.csv', lable_column='category_id'):
        self.annotation_path = annotation_path
        self.train_features = []
        self.train_labels = []
        self.val_features = []
        self.val_labels = []
        self.prepare(lable_column=lable_column)

    def prepare(self, lable_column):
        df = pd.read_csv(self.annotation_path)
        df['path'] = df.product_id.apply(lambda x: os.path.join(DATASET_DIR, x + ".jpg"))
        for class_index in class_mapping.keys():
            data = df[df['category_id'] == class_index]
            np.random.seed(0)
            mask = np.random.rand(len(data)) < 0.8
            self.train_features.extend(data[mask]["path"].tolist())
            self.train_labels.extend(data[mask][lable_column].tolist())
            self.val_features.extend(data[~mask]["path"].tolist())
            self.val_labels.extend(data[~mask][lable_column].tolist())
        print(f"There are {len(self.train_features)} training images, {len(self.val_features)} validation images")
        # print(f"Training images class distribution {len(self.train)} , {len(self.val)} validation images")

    def build(self, mode="train"):
        if mode == "train":
            dataset = tf.data.Dataset.from_tensor_slices(
                (self.train_features, self.train_labels))
        elif mode == "valid":
            dataset = tf.data.Dataset.from_tensor_slices(
                (self.val_features, self.val_labels))
        dataset = dataset.map(load_from_path_label, num_parallel_calls=AUTOTUNE)
        if mode == "train":
            dataset = dataset.map(random_rotate, num_parallel_calls=AUTOTUNE)
            dataset = dataset.map(random_flip, num_parallel_calls=AUTOTUNE)
            dataset = dataset.shuffle(
                buffer_size=1000, reshuffle_each_iteration=True).repeat(count=STEPS).batch(BATCH)
        elif mode == "valid":
            dataset = dataset.shuffle(
                buffer_size=1000, reshuffle_each_iteration=True).batch(BATCH)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset


def load_from_path_label(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [input_width, input_height])
    # image_normalize = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)(image)
    # tf.print(image.shape)
    return tf.cast(image, tf.float64), tf.cast(label-1, tf.int32)


def random_rotate(image, label):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(image, rn), label


def random_flip(image, label):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (image, label),
                   lambda: (tf.image.flip_left_right(image), label))