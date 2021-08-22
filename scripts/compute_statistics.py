import os
import pandas as pd
from imageio import imread
from matplotlib import pyplot as plt
import seaborn as sns
from config import DATASET_DIR, class_mapping

def _build_and_save_count_chart(figure_name, output_dir, count):
    fig, axs = plt.subplots()
    sns.countplot(x=count, ax=axs)
    fig_filepath = os.path.join(output_dir, figure_name) + ".png"
    axs.set_ylabel('count')
    plt.xticks(rotation=90)
    axs.set_title(figure_name)
    fig.savefig(fig_filepath)
    plt.close()

df = pd.read_csv("../data/train.csv")
print(f'Total {df.shape[0]} images in annotation csv file, including {df.shape[1]} columns: {df.columns.to_list()}')
df['path'] = df.product_id.apply(lambda x: os.path.join(DATASET_DIR, x + ".jpg"))
df['imshape'] = df.path.apply(lambda x: imread(x).shape[:2])
df['height'] = df.imshape.apply(lambda x: x[0])
df['width'] = df.imshape.apply(lambda x: x[1])
df['pixel_size'] = df['height']*df['width']
df['category'] = df['category_id'].apply(lambda x: class_mapping.get(x, "unkown"))
print(df.head())

def move_file_to_folder():
    for path, label in zip(df['path'].to_list(), df['category_id'].to_list()):
        src = path
        new_dir = "./images/train_class"
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        if not os.path.exists(os.path.join(new_dir, str(label))):
            os.makedirs(os.path.join(new_dir, str(label)))
        dst = os.path.join(new_dir, str(label), os.path.basename(path))
        from shutil import copyfile
        print("src", src, "dst", dst)
        copyfile(src, dst)
move_file_to_folder()

_build_and_save_count_chart("cat_count", "../data", df['category'])

fig, ax = plt.subplots()
df.hist('height', ax=ax)
fig.savefig('../data/hist_h.png')
fig, ax = plt.subplots()
df.hist('width', ax=ax)
fig.savefig('../data/hist_w.png')
fig = plt.figure()
plt.scatter(df['width'], df['height'], marker="x")
plt.xlabel('width')
plt.xlabel('height')
fig.savefig('../data/w_h.png')
# Images Example
train_images = [f for f in os.listdir(DATASET_DIR) if os.path.isfile(os.path.join(DATASET_DIR, f))]
print(f'Total {len(train_images)} images in train folder. 5 Training images', train_images[:5])

# fig = plt.figure(figsize=(15, 10))
# columns = 5; rows = 4
# for i in range(1, columns*rows+1):
#     ds = imread(df.iloc[i]['path'])
#     fig.add_subplot(rows, columns, i)
#     plt.imshow(ds)

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