import os
import matplotlib.pyplot as plt

from PIL import Image
import cv2

from torch.utils.data import Dataset, DataLoader
from transforms import Transforms, NormTransforms

class PlushieNormDataset(Dataset):

    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = [entry for entry in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, entry))]
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img = cv2.imread(os.path.join(self.img_dir, self.samples[i]))
        img = self.transform(img)
        return img
    


class PlushieTrainDataset(Dataset):
    
    def __init__(self, filepath, img_dir, transform=None):
        self.samples = []
        self.img_dir = img_dir
        self.transform = transform
        

        with open(filepath, 'r') as f:
            self.samples = [line.strip() for line in f]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        line = self.samples[i].split()
        if len(line) == 3:
            anchor_name, anchor_num, img_num = line
            img_name = anchor_name
            is_same = 1
        elif len(line) == 4:
            anchor_name, anchor_num, img_name, img_num = line
            is_same = 0
        else:
            print(len(line), line)
            raise Exception("Shouldn't be here")
        
        anchor = cv2.imread(os.path.join(self.img_dir, f"{anchor_name}_{anchor_num}.png"))
        img = cv2.imread(os.path.join(self.img_dir, f"{img_name}_{img_num}.png"))
        
        if self.transform:
            anchor = self.transform(anchor)
            img = self.transform(img)

        return anchor, img, is_same

def create_norm_dataset(filepath, img_dir):
    t = NormTransforms()
    d = PlushieNormDataset(img_dir=img_dir, transform=t)
    loader = DataLoader(d, batch_size=len(d), num_workers=1)
    data = next(iter(loader))
    mean = [data[:, 0].mean(), data[:, 1].mean(), data[:, 2].mean()]
    std = [data[:, 0].std(), data[:, 1].std(), data[:, 2].std()]
    
    t_norm = Transforms(mean=mean, std=std)
    d_norm = PlushieTrainDataset(filepath=filepath, img_dir=img_dir, transform=t_norm)
    return d_norm

def main():
    filepath = r"/content/OP_ReID_GPTEAM/Datasets/Processed/train_ann_50.txt"
    img_dir = r"/content/OP_ReID_GPTEAM/Datasets/Processed/train_images"

    d_norm = create_norm_dataset(filepath=filepath, img_dir=img_dir)
    
    e = d_norm[0]
    axs = plt.figure(figsize=(9, 9)).subplots(1, 2)
    plt.title(e[2])
    axs[0].imshow(e[0].permute(1,2,0))
    axs[1].imshow(e[1].permute(1,2,0))

if __name__ == "__main__":
    main()
