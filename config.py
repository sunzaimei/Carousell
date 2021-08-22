import os

BATCH = 32  # 128
STEPS = 3000
# LEARNING_RATE = 0.0001
DATASET_DIR = '../images/train'
EVAL_DIR = '../images/eval-category'
LOG_DIR = '../weights'
class_no = 12
input_width = 320
input_height = 320
# checkpoint_dir = None
checkpoint_dir = os.path.join(LOG_DIR, '20210822-024844')
class_mapping = {
    1: "Laptops & Notebooks",
    2: "Desktops",
    3: "Cables & Adaptors",
    4: "Chargers",
    5: "Mouse & Mousepads",
    6: "Monitor Screens",
    7: "Computer Keyboard",
    8: "Hard Disks & Thumbdrives",
    9: "Networking Parts & Accessories",
    10: "Webcams",
    11: "Laptop Bags & Sleeves",
    12: "Printers, Scanners & Copiers"}
