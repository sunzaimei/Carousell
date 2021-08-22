import math

import numpy as np
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import argparse

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.python.keras.backend import categorical_crossentropy
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import PiecewiseConstantDecay

from model.evaluation import CatEvaluation, PriceEvaluation

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

parser = argparse.ArgumentParser(description='check if got existing checpoint to resume from')
parser.add_argument('--checkpoint', default=None)
args = parser.parse_args()
print(args)


# config
CHECKPOINT = args.checkpoint
# CHECKPOINT = './model_trained/model.ckpt-30000'

import os
import time
import tensorflow as tf
# from model.model import evaluate
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam


class Trainer:
    def __init__(self,
                 model,
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                 learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5]),
                 checkpoint_dir='', class_mapping={}, metrics_name='f1'):

        self.now = None
        self.loss = loss
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0), score=tf.Variable(-1.0),
                                              optimizer=Adam(learning_rate=learning_rate), model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=5)
        self.class_mapping = class_mapping
        self.metrics_name = metrics_name
        self.restore()
        train_log_dir = os.path.join(checkpoint_dir, 'gradient_tape/train')
        eval_log_dir = os.path.join(checkpoint_dir, 'gradient_tape/eval')
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)
        # model.summary()

    @property
    def model(self):
        return self.checkpoint.model

    def train(self, train_dataset, valid_dataset, steps, evaluate_every=100, save_best_only=False):
        """
        :param train_dataset:
        :param valid_dataset:
        :param steps: total 5000 epochs with 100 iterations each
        :param evaluate_every: save checkpoint every epoch, which is 100 iterations
        :param save_best_only: Save a checkpoint only if evaluation PSNR has improved, thus remaining to be the best known
        :return:
        """
        loss_mean = Mean()
        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint
        self.now = time.perf_counter()
        tf.print("steps", steps, ckpt.step.numpy())
        for image, label in train_dataset.take(steps - ckpt.step.numpy()):
            ckpt.step.assign_add(1)
            step = ckpt.step.numpy()
            loss = self.train_step(image, label)
            duration = time.perf_counter() - self.now
            loss_mean(loss)
            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', loss_mean.result(), step=step)

            if step % 106 == 0:
                loss_value = loss_mean.result()
                loss_mean.reset_states()
                print(f'Epoch {int(step / 106)}/{int(steps / 106)} - {step}/{steps}: loss = {loss_value:.6f}')
            if step % evaluate_every == 0:
                # Compute metrics on validation dataset
                score, runtime_value = self.evaluate(valid_dataset)
                score_train, runtime_value_train = self.evaluate(train_dataset.take(100))
                print(f'{self.metrics_name}_train ={score_train:3f}, {self.metrics_name} = {score:3f} ({duration:.2f}s), '
                      f'runtime_value = {runtime_value:3f}')

                with self.train_summary_writer.as_default():
                    tf.summary.scalar(self.metrics_name, score_train, step=step / 106)
                with self.eval_summary_writer.as_default():
                    tf.summary.scalar(self.metrics_name, score, step=step/106)
                    tf.summary.scalar('runtime', runtime_value, step=step/106)

                if save_best_only and score <= ckpt.score:
                    self.now = time.perf_counter()
                    # skip saving checkpoint, no PSNR improvement
                    continue

                ckpt.score = score
                ckpt_mgr.save()

                self.now = time.perf_counter()

    @tf.function
    def train_step(self, image, label):
        with tf.GradientTape() as tape:
            # image = tf.cast(image, tf.float32)
            # label = tf.cast(label, tf.int32)
            prediction = self.checkpoint.model(image) # is_training
            loss_value = self.loss(label, prediction)
            # tf.print("loss_value", loss_value)
            # if tf.math.is_nan(loss_value):
            # tf.print("prediction", tf.where(label==12), tf.shape(prediction))

        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))
        return loss_value

    def evaluate(self, dataset, save_result=False, checkpoint_dir='', post_process=False):
        model = self.checkpoint.model
        runtime_values = []
        evaluator = CatEvaluation(num_classes=len(self.class_mapping.keys()), class_mapping=self.class_mapping)
        for image, labels in dataset:
            start_time = time.perf_counter()
            prediction = model(image)
            runtime = (time.perf_counter() - start_time) * 1000
            results = tf.math.argmax(prediction, axis=-1)
            evaluator.add_single_result(labels, results)
            runtime_values.append(runtime)
        overall_f1_score = evaluator.evaluate()
        avg_runtime = tf.reduce_mean(runtime_values[1:]) if len(runtime_values) > 1 else tf.reduce_mean(
            runtime_values)
        return overall_f1_score, avg_runtime

    def predict(self, dataset, save_result=False, checkpoint_dir='', post_process=False):
        # /online code
        model = self.checkpoint.model
        runtime_values = []
        for image, labels in dataset:
            start_time = time.perf_counter()
            prediction = model(image)
            runtime = (time.perf_counter() - start_time) * 1000
            results = tf.math.argmax(prediction, axis=-1)
            # print("results", tf.shape(results), "labels", tf.shape(labels))
            runtime_values.append(runtime)
        avg_runtime = tf.reduce_mean(runtime_values[1:]) if len(runtime_values) > 1 else tf.reduce_mean(
            runtime_values)
        return avg_runtime

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint).expect_partial()
            print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')


class PriceTrainer(Trainer):
    def evaluate(self, dataset, save_result=False, checkpoint_dir='', post_process=False):
        model = self.checkpoint.model
        runtime_values = []
        evaluator = PriceEvaluation()
        for image, labels in dataset:
            start_time = time.perf_counter()
            predictions = model(image)
            predictions = tf.squeeze(predictions)
            runtime = (time.perf_counter() - start_time) * 1000
            evaluator.add_single_result(labels, predictions)
            runtime_values.append(runtime)
        mae_score = evaluator.evaluate()
        avg_runtime = tf.reduce_mean(runtime_values[1:]) if len(runtime_values) > 1 else tf.reduce_mean(
            runtime_values)
        return mae_score, avg_runtime

    def predict(self, dataset, save_result=False, checkpoint_dir='', post_process=False):
        # /online code
        model = self.checkpoint.model
        runtime_values = []
        for image, labels in dataset:
            start_time = time.perf_counter()
            prediction = model(image)
            runtime = (time.perf_counter() - start_time) * 1000
            runtime_values.append(runtime)
        avg_runtime = tf.reduce_mean(runtime_values[1:]) if len(runtime_values) > 1 else tf.reduce_mean(
            runtime_values)
        return avg_runtime




