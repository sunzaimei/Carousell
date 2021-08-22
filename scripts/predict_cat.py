import pandas as pd
import tensorflow as tf
import numpy as np
import os
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import PiecewiseConstantDecay
from model.trainer import Trainer
from config_cat import class_no, input_width, input_height, EVAL_DIR, class_mapping, checkpoint_dir
from tensorflow.keras.preprocessing import image

# define model here
input_shape = (input_height, input_width, 3)
base_model = ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(class_no, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
# trainer = Trainer(model=model, checkpoint_dir=checkpoint_dir, class_mapping=class_mapping,
#                   learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5]))
trainer = Trainer(model=model, checkpoint_dir=checkpoint_dir, class_mapping=class_mapping,
                  learning_rate=PiecewiseConstantDecay(boundaries=[20000], values=[5e-5, 1e-6]),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics_name="f1")
print(f'Model will be stored in checkpoint_dir: {checkpoint_dir}')

trainer.restore()
predicter = trainer.checkpoint.model
# # Evaluate model on full validation set.
eval_df = pd.read_csv("submission/eval-category_sample.csv")
original_columns = eval_df.columns.to_list()
print(eval_df.head())
eval_df['path'] = eval_df.product_id.apply(lambda x: os.path.join(EVAL_DIR, x + ".jpg"))
img_paths = eval_df['path'].to_list()
results = []
for img_path in img_paths:
    img = image.load_img(img_path, target_size=(input_height, input_width))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = predicter.predict(x)
    result = tf.math.argmax(preds, axis=-1)
    results.append(result[0].numpy())
eval_df["category_id"] = results
eval_df[original_columns].to_csv("submission/eval-category.csv", index=False)
print(eval_df.head())




