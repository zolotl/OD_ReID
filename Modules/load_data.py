import os, glob
import random
import numpy as np
from PIL import Image
import json
import cv2

def process_image(img_file, shift_size=20):
    img = cv2.imread(img_file, cv2.COLOR_BayerBG2RGB)
    img = cv2.rotate(img, 0)
    shift = np.random.randint(shift_size, size=(1, 1, 3))
    img = img + shift
    img = np.mod(img, 256)

def crop_image(img_dir, labels_dir, cropped_img_dir, img_map, img_map_json):
    count = 0
    for filename in os.listdir(labels_dir):
        labels_path = os.path.join(labels_dir, filename)
        img_path = os.path.join(img_dir, filename.replace('txt', 'png'))
        # checking if it is a file
        if os.path.isfile(labels_path):
            with open(labels_path, 'r') as f:
                lines = [line.strip() for line in f.readlines()]
                for line in lines:
                    if line is not '':
                        p_id, x_center, y_center, width, height = [float(x) for x in line.split()]
                        im = Image.open(img_path)
                        im = im.resize((600, 600))
                        if p_id in img_map.keys():
                            img_map[p_id] += 1
                        else:
                            img_map[p_id] = 1
                        im_width = 600
                        im_height = 600
                        im = im.crop(((x_center-width/2)*im_width, (y_center-height/2)*im_height, (x_center+width/2)*im_width, (y_center+height/2)*im_height))
                        im.save(os.path.join(cropped_img_dir, f"{p_id}_{img_map[p_id]}.png"))
        if (count * 100 /round(len(os.listdir(labels_dir)), -2)) % 1 == 0:
            print(count * 100 /round(len(os.listdir(labels_dir)), -2))
        count += 1
    with open(img_map_json, 'w') as fp:
        json.dump(img_map, fp)
    

def create_data(ann_file, img_map):     
    with open(ann_file, 'w') as f:      
        for anchor_class in img_map.keys():
            # check if there is more than 1 of the same plush
            anchor_nums = img_map[anchor_class]
            if anchor_nums > 1:
                # Make anchor positive pair   
                for _ in range(50): 
                    anchor_num = random.randint(1, anchor_nums)     
                    positive_num = random.randint(1, anchor_nums)
                    while positive_num == anchor_num:
                        positive_num = random.randint(1, anchor_nums)
                    f.write(f'{anchor_class} {anchor_num} {positive_num}' + '\n')
            # Make anchor negative pair
            for _ in range(50):
                anchor_num = random.randint(1, anchor_nums)
                neg_class = random.choice(list(img_map.keys()))
                while neg_class == anchor_class:
                    neg_class = random.choice(list(img_map.keys()))
                neg_num = random.randint(1, img_map[neg_class])
                f.write(f'{anchor_class} {anchor_num} {neg_class} {neg_num}' + '\n')