# OP_ReID_GPTEAM

## Setting up environment in Google Colab
### Clone Repo and install dependencies
```shell
!git clone https://github.com/zolotl/OP_ReID_GPTEAM.git
%cd OP_ReID_GPTEAM/
!pip install requirements.txt
%cd ../
```

### Upload datasets
Create required dataset directories
```shell
%cd OP_ReID_GPTEAM/

!mkdir Datasets
!mkdir Datasets/Raw
!mkdir Datasets/Raw/train
!mkdir Datasets/Raw/train_labels
!mkdir Datasets/Raw/val
!mkdir Datasets/Raw/val_labels
!mkdir Datasets/Raw/suspects
!mkdir Datasets/Raw/test
!mkdir Datasets/Processed
!mkdir Datasets/Processed/train_images
!mkdir Datasets/Processed/val_images
```
Fetch datasets from google drive folder
```shell
!gdown --folder 1-gkyVhq_9bUrtlAPH2dF9hKY5RHlUvj_
#possible folder options: 
  #1-g0AtQppS6ZZH3I2xXOGPZCb1eoY2lLR
  #1-gkyVhq_9bUrtlAPH2dF9hKY5RHlUvj_
  #10aQzclJMpk-N_RQgr42k41HdPhjdZFvf
  #108mLafHSc4kpPdD7C2yyt7YkRhhW4yOU
```

Unzip datasets into these directories
```shell
!unzip /content/Novice/CV/Train.zip -d /content/OP_ReID_GPTEAM/Datasets/Raw/train
!unzip /content/Novice/CV/train_labels.zip -d /content/OP_ReID_GPTEAM/Datasets/Raw/train_labels
!unzip /content/Novice/CV/Validation.zip -d /content/OP_ReID_GPTEAM/Datasets/Raw/val
!unzip /content/Novice/CV/val_labels.zip -d /content/OP_ReID_GPTEAM/Datasets/Raw/val_labels
!unzip /content/Novice/CV/suspects.zip -d /content/OP_ReID_GPTEAM/Datasets/Raw/suspects
!unzip /content/Novice/CV/Test.zip -d /content/OP_ReID_GPTEAM/Datasets/Raw/test
```

### Process siamese model dataset
```shell
!python /content/OP_ReID_GPTEAM/Modules/create_data.py
```

### Clone yolov5 repo
```shell
!git clone https://github.com/ultralytics/yolov5  # clone
%cd yolov5
!pip install -r requirements.txt  # install
%cd ../
```

### Train siamese model
```shell
!python /content/OP_ReID_GPTEAM/Modules/train.py
```

### Generate predictions using trained model
```shell
!python /content/OP_ReID_GPTEAM/Modules/inference.py
```
