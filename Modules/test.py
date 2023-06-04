import matplotlib.pyplot as plt

from PIL import Image
import cv2

import torch

from model import SiameseNetwork
from my_utils import get_default_device, show_img, to_device
from transforms import Transforms

yolo_model = torch.hub.load('../yolov5', 'custom', path='best_new.pt', source='local')
yolo_model.iou = 0.05
yolo_model.conf = 0.4


def predict_image(model_path, target, img, transform=None):
    xb, xb2 = transform(target).unsqueeze(0), transform(img).unsqueeze(0) # Convert to batch of 1
    device = get_default_device()
    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
    to_device(model, device)
    model.eval()
    yb = model(xb.to(device), xb2.to(device))
    return yb

device = get_default_device()
model = SiameseNetwork()
model.load_state_dict(torch.load('../Output/model_2.pth', map_location=torch.device('cuda')))
to_device(model, device)

t = Transforms()

target = r"../Datasets/Raw/suspects/image_0523.png"
img = r"../Datasets/Raw/test/image_0523.png"

target = cv2.imread(target)
img = cv2.imread(img)

# Perform object detection
results = yolo_model(img)
for result in results.xyxy[0].tolist():
  x1, y1, x2, y2 = [int(x) for x in result[:4]]
  plushie = img[y1:y2, x1:x2]
  print(x1, ', ', y1, ', ', x2, ', ', y2)
  print(y1/img.shape[0], x1/img.shape[1], y2/img.shape[0], x2/img.shape[1])
  print(float(predict_image('../Output/model_2.pth', target, plushie, transform=t))) # > 0 means match, < 0 means no match
axs = plt.figure(figsize=(9, 9)).subplots(1, 2)
axs[0].imshow(target)
axs[1].imshow(img)



# -3.3546

