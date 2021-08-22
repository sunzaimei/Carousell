import os
from model.dataset import PriceDataset

lable_column = "price"
BATCH = 32  # 128
STEPS = 3000
# LEARNING_RATE = 0.0001
DATASET_DIR = '../images/train'
EVAL_DIR = '../images/eval-price'
LOG_DIR = '../weights'
class_no = 12
input_width = 320
input_height = 320
checkpoint_dir = None
# checkpoint_dir = os.path.join(LOG_DIR, 'price_20210822-140424')
# checkpoint_dir = os.path.join(LOG_DIR, 'price_20210822-140712')
dataset_loader = PriceDataset(input_width, input_height, DATASET_DIR, annotation_path='../data/train.csv',
                              lable_column=lable_column)
