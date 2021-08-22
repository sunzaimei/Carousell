import os

import numpy as np
import pandas as pd
from imageio import imread
from matplotlib import pyplot as plt
import seaborn as sns
from config_cat import DATASET_DIR, class_mapping
from shutil import copyfile


def build_and_save_count_chart(figure_name, output_dir, count):
    fig, axs = plt.subplots()
    from matplotlib.pyplot import figure
    # figure(figsize=(8, 6), dpi=80)
    sns.countplot(x=count, ax=axs)
    fig_filepath = os.path.join(output_dir, figure_name) + ".png"
    axs.set_ylabel('count')
    plt.xticks(rotation=45)
    axs.set_title(figure_name)
    fig.savefig(fig_filepath)
    plt.close()


def build_and_save_hist_chart(df, column, bins, save_folder, name):
    fig, ax = plt.subplots()
    df.hist(column, bins=bins, ax=ax)
    fig.savefig(os.path.join(save_folder, name))


def move_file_to_folder(df, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for path, label in zip(df['path'].to_list(), df['category_id'].to_list()):
        src = path
        dst_sub_dir = os.path.join(dst_dir, str(label))
        if not os.path.exists(dst_sub_dir):
            os.makedirs(dst_sub_dir)
        dst = os.path.join(dst_sub_dir, os.path.basename(path))
        print(f"move from {src} to {dst}")
        copyfile(src, dst)


if __name__ == "__main__":
    df = pd.read_csv("../data/train.csv")
    print(f'Total {df.shape[0]} images in annotation csv file, including {df.shape[1]} columns: {df.columns.to_list()}')
    df['path'] = df.product_id.apply(lambda x: os.path.join(DATASET_DIR, x + ".jpg"))
    df['price'] = df.price.apply(lambda x: x.replace(",", ""))
    df['price'] = df.price.astype(float)
    df['imshape'] = df.path.apply(lambda x: imread(x).shape[:2])
    df['height'] = df.imshape.apply(lambda x: x[0])
    df['width'] = df.imshape.apply(lambda x: x[1])
    df['pixel_size'] = df['height']*df['width']
    df['category'] = df['category_id'].apply(lambda x: class_mapping.get(x, "unkown"))
    print(df.head())
    move_file_to_folder(df, dst_dir="../images/train_class")  # do this for manual examination of labels
    # build charts
    save_folder = "../data/statistics"
    build_and_save_count_chart("cat_count", save_folder, df['category_id'])
    build_and_save_hist_chart(df, "height", 20, save_folder, 'hist_h.png')
    build_and_save_hist_chart(df, "width", 20, save_folder, 'hist_w.png')
    build_and_save_hist_chart(df, "pixel_size", 20, save_folder, 'hist_pixel.png')
    # scatter plot for image shapes, width vs height
    fig = plt.figure()
    plt.scatter(df['width'], df['height'], marker="x")
    plt.xlabel('width')
    plt.ylabel('height')
    fig.savefig(os.path.join(save_folder, 'w_h.png'))
    # scatter plot for image price for different categories
    fig = plt.figure()
    plt.scatter(df['category_id'], df['price'], marker="x")
    plt.xlabel('category')
    plt.xlabel('category_id')
    plt.ylabel('price')
    plt.xticks(np.arange(1, 13, step=1))
    fig.savefig(os.path.join(save_folder, 'cat_price.png'))

    # Images Example
    train_images = [f for f in os.listdir(DATASET_DIR) if os.path.isfile(os.path.join(DATASET_DIR, f))]
    print(f'Total {len(train_images)} images in train folder. 5 Training images', train_images[:5])

# model = LeNet()
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
# dataset_loader = CarouselDataset(annotation_path='./data/train.csv')
# train_ds = dataset_loader.build(mode='train')
# valid_ds = dataset_loader.build(mode='valid')
# history = model.fit(
#   train_ds,
#   validation_data=valid_ds,
#   epochs=50
# )