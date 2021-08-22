import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.data import AUTOTUNE


class CatDataset:
    def __init__(self, input_width, input_height, dataset_dir, annotation_path='./data/train.csv', lable_column='category_id',
                 class_mapping=None):
        self.annotation_path = annotation_path
        self.train_features = []
        self.train_labels = []
        self.val_features = []
        self.val_labels = []
        self.input_width = input_width
        self.input_height = input_height
        self.lable_column = lable_column
        self.class_mapping = class_mapping
        self.dataset_dir = dataset_dir
        self.prepare(lable_column=lable_column)

    def prepare(self, lable_column):
        df = pd.read_csv(self.annotation_path)
        df['path'] = df.product_id.apply(lambda x: os.path.join(self.dataset_dir, x + ".jpg"))
        for class_index in self.class_mapping.keys():
            data = df[df[lable_column] == class_index]
            np.random.seed(0)
            mask = np.random.rand(len(data)) < 0.8
            self.train_features.extend(data[mask]["path"].tolist())
            self.train_labels.extend(data[mask][lable_column].tolist())
            self.val_features.extend(data[~mask]["path"].tolist())
            self.val_labels.extend(data[~mask][lable_column].tolist())

        print(f"There are {len(self.train_features)} training images, {len(self.val_features)} validation images")

    def build(self, mode, batch_size, count=-1):
        if mode == "train":
            dataset = tf.data.Dataset.from_tensor_slices(
                (self.train_features, self.train_labels))
        elif mode == "valid":
            dataset = tf.data.Dataset.from_tensor_slices(
                (self.val_features, self.val_labels))
        dataset = dataset.map(self.load_from_path_label, num_parallel_calls=AUTOTUNE)
        if mode == "train":
            dataset = dataset.map(random_rotate, num_parallel_calls=AUTOTUNE)
            dataset = dataset.map(random_flip, num_parallel_calls=AUTOTUNE)
            dataset = dataset.shuffle(
                buffer_size=1000, reshuffle_each_iteration=True).repeat(count=count).batch(batch_size)
        elif mode == "valid":
            dataset = dataset.shuffle(
                buffer_size=1000, reshuffle_each_iteration=True).batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset

    def load_from_path_label(self, path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [self.input_width, self.input_height])
        return tf.cast(image, tf.float64), tf.cast(label-1, tf.int32)


class PriceDataset(CatDataset):
    def prepare(self, lable_column):
        df = pd.read_csv(self.annotation_path)
        df['price'] = df['price'].apply(lambda x: x.replace(",", ""))
        df['price'] = df['price'].astype(float)
        print(df.price.dtype)
        df['path'] = df.product_id.apply(lambda x: os.path.join(self.dataset_dir, x + ".jpg"))
        np.random.seed(0)
        mask = np.random.rand(len(df)) < 0.8
        self.train_features.extend(df[mask]["path"].tolist())
        self.train_labels.extend(df[mask][lable_column].tolist())
        self.val_features.extend(df[~mask]["path"].tolist())
        self.val_labels.extend(df[~mask][lable_column].tolist())
        print(f"There are {len(self.train_features)} training images, {len(self.val_features)} validation images")

    def load_from_path_label(self, path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [self.input_width, self.input_height])
        return tf.cast(image, tf.float64), tf.cast(label, tf.float32)


def random_rotate(image, label):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(image, rn), label


def random_flip(image, label):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (image, label),
                   lambda: (tf.image.flip_left_right(image), label))