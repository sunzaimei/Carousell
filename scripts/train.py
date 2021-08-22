import datetime
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import PiecewiseConstantDecay

from model.dataset import CarouselDataset
from model.model import LeNet, lenet, simpleNet

from model.trainer import Trainer
import os
from config import STEPS, LOG_DIR, class_no, input_width, input_height
from config import class_mapping, checkpoint_dir

dataset_loader = CarouselDataset(annotation_path='../data/train.csv', lable_column='category_id')
train_ds = dataset_loader.build(mode='train')
valid_ds = dataset_loader.build(mode='valid')

if checkpoint_dir is None:
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = os.path.join(LOG_DIR, current_time)  # add model name

# define model here
input_shape = (input_width, input_height, 3)
# base_model = InceptionV3(input_shape=input_shape, include_top=False, weights='imagenet')
base_model = ResNet50(input_shape=input_shape, include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x, training=True)
predictions = Dense(class_no, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers[:]:
    layer.trainable = False

# model = lenet(is_training=True)
# model = simpleNet()
# model.summary()
trainer = Trainer(model=model, checkpoint_dir=checkpoint_dir, class_mapping=class_mapping,
                  # learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5]),
                  learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[5e-5, 1e-6]),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy())
print(f'Model will be stored in checkpoint_dir: {checkpoint_dir}')
trainer.restore()
trainer.train(train_ds, valid_ds, steps=STEPS*106, evaluate_every=106*10, save_best_only=True)
trainer.restore()
# # Evaluate model on full validation set.
f1_score, runtime = trainer.evaluate(valid_ds)
print(f'f1_score = {f1_score.numpy():3f}, RUNTIME = {runtime:3f}')
#
# # Save model
# saved_model_path = os.path.join(checkpoint_dir, 'saved_models', str(int(time.time())))
# tf.saved_model.save(trainer.model, saved_model_path)
# # converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
#
# # Save h5
# keras_model_file = os.path.join(checkpoint_dir, 'saved_models', 'model.h5')
# trainer.model.save(keras_model_file)


