import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import PiecewiseConstantDecay
from model.trainer import PriceTrainer
from config_price import input_width, input_height, EVAL_DIR, checkpoint_dir
from tensorflow.keras.preprocessing import image

# define model here
input_shape = (input_height, input_width, 3)
base_model = ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation="relu")(x)
model = Model(inputs=base_model.input, outputs=predictions)
# model = LeNet()
trainer = PriceTrainer(model=model, checkpoint_dir=checkpoint_dir, class_mapping={},
                       # learning_rate=PiecewiseConstantDecay(boundaries=[20000, values=[1e-4, 5e-5]),
                       learning_rate=PiecewiseConstantDecay(boundaries=[20000], values=[5e-5, 1e-6]),
                       loss=tf.keras.losses.MeanSquaredError(), metrics_name="mae")

print(f'Model will be stored in checkpoint_dir: {checkpoint_dir}')
trainer.restore()
predicter = trainer.checkpoint.model

# Evaluate model on full validation set.
eval_df = pd.read_csv("../submission/eval-price_sample.csv")
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
    results.append(int(preds[0][0]))
eval_df["price"] = results
eval_df[original_columns].to_csv("../submission/eval-price.csv", index=False)
print(eval_df.head())




