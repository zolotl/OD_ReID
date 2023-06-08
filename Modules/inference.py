
from contextlib import nullcontext
import torch
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import os

from test import predict_image
from transforms import Transforms

yolo_model = torch.hub.load('../yolov5', 'custom', path='./best_new.pt', source='local')
yolo_model.iou = 0.01
yolo_model.conf = 0.4
reid_model = "./model_2.pth"

reid_transforms = Transforms()

def detect_objects(image_id, image_path, suspect_path, results_arr = []):
    # Read image
    img = cv2.imread(image_path)

    # Perform object detection
    results = yolo_model(img)
    results = results.xyxy[0].tolist()

    # Restrict the number of predictions to 4 per image
    if len(results) > 4:
      results = results[:4]
    
    # Check is yolo prediction is null
    if len(results) < 1:
      results_arr.append([image_id, 0, 0, 0, 0, 0, 0])

    # Store detected objects with their classes, confidence scores, and bounding box coordinates in results array
    for result in results:
      confidence = result[4]
      x1, y1, x2, y2 = [int(x) for x in result[:4]]
      plushie = img[y1:y2, x1:x2]
      suspect = cv2.imread(suspect_path)
      match_confidence = float(predict_image(reid_model, suspect, plushie, transform=reid_transforms))
      plushie_class = 1 if match_confidence > 0 else 0
      results_arr.append([image_id, plushie_class, confidence, y1/img.shape[0], x1/img.shape[1], y2/img.shape[0], x2/img.shape[1]])

    
# test and suspect images directory paths 
test_dir = r'../Datasets/Raw/test'
sus_dir = r'../Datasets/Raw/suspects/content/drive/Shareddrives/ZINDI Data Science/ADPL/Competition Data/CV/Data Prep/Test (0-1599)/merged/crops'
results = []
count = 0


# Iterate directory
for path in os.listdir(test_dir):
    # check if current path is a file
    test_path = os.path.join(test_dir, path)
    sus_path = os.path.join(sus_dir, path)
    if os.path.isfile(test_path):
      detect_objects(path.removesuffix('.png'), test_path, sus_path, results)
      count += 1
    percentage = count / round(len(os.listdir(test_dir)), -2) * 100
    print(count)
    if percentage % 5 == 0:
      headers = ['Image_ID', 'class', 'confidence', 'ymin', 'xmin', 'ymax', 'xmax']
      print(len(results))
      results_df = pd.DataFrame(results, columns=headers)
      results_df.to_csv('../Output/Results/results_2.csv', index=False)
      print(percentage)


headers = ['Image_ID', 'class', 'confidence', 'ymin', 'xmin', 'ymax', 'xmax']
results_df = pd.DataFrame(results, columns=headers)
results_df.to_csv('../Output/Results/results_2.csv', index=False)
