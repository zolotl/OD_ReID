from load_data import crop_image, create_data
import json

train_map_json = '/content/OP_ReID_GPTEAM/Modules/train_map.json'
val_map_json = '/content/OP_ReID_GPTEAM/Modules/val_map.json'

with open(train_map_json, 'r') as fp:
    train_img_map = json.load(fp)
with open(val_map_json, 'r') as fp:
    val_img_map = json.load(fp)

val_img_path = r"/content/OP_ReID_GPTEAM/Datasets/Raw/val"
val_ann_path = r"/content/OP_ReID_GPTEAM/Datasets/Raw/val_labels"
val_cropped_img_dir = r"/content/OP_ReID_GPTEAM/Datasets/Processed/val_images"

crop_image(val_img_path, val_ann_path, val_cropped_img_dir, val_img_map, val_map_json)
create_data(r'/content/OP_ReID_GPTEAM/Datasets/Processed/val_ann_50.txt', val_img_map)

train_img_path = r"/content/OP_ReID_GPTEAM/Datasets/Raw/train"
train_ann_path = r"/content/OP_ReID_GPTEAM/Datasets/Raw/train_labels"
train_cropped_img_dir = r"/content/OP_ReID_GPTEAM/Datasets/Processed/train_images"

crop_image(train_img_path, train_ann_path, train_cropped_img_dir, train_img_map, train_map_json)
create_data(r'/content/OP_ReID_GPTEAM/Datasets/Processed/train_ann_50.txt', train_img_map)
