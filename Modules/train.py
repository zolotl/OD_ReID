import numpy as np

import torch
import sys

from tqdm import tqdm

from dataset import PlushieTrainDataset
from model import SiameseNetwork
from transforms import Transforms, Transforms_val
from my_utils import DeviceDataLoader, accuracy, get_default_device, to_device



def loss_batch(model, loss_func, anchor, image, label, opt=None, metric=None): # Update model weights and return metrics given xb, yb, model
    preds = model(anchor, image)
    loss = loss_func(preds, label.unsqueeze(1).float())
    
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
        
    metric_result = None
    if metric is not None:
        metric_result = metric(preds, label)
        
    return loss.item(), len(anchor), metric_result


def fit(num_epochs, model, loss_func, train_dl, val_dl, opt_func=torch.optim.SGD, lr=0.01, metric=None, checkpoint = None):
    train_losses, val_losses, val_metrics = [] , [], []
    
    opt = opt_func(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=2, threshold=5e-3, verbose=True)

    if checkpoint != None:
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        current_epoch = checkpoint['epoch']
    else:
        current_epoch = 1
    
    for epoch in range(current_epoch, num_epochs+1):
        model.train() # Setting for pytorch - training mode
        for anchor,image,label in tqdm(train_dl):
            train_loss, _, _ = loss_batch(model, loss_func, anchor, image, label, opt) # update weights
            
        model.eval() # Setting - eval mode
        val_loss, total, val_metric = evaluate(model, loss_func, val_dl, metric)
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_metrics.append(val_metric)
        
        if metric is None:
            print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}".format(
            epoch, train_loss, val_loss))
        else:
            print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_{}: {:.4f}".format(
            epoch, train_loss, val_loss, metric.__name__, val_metric))

        # Save model checkpoints
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            metric.__name__: val_metric
            }, '/content/drive/MyDrive/results_cv/model3_{}_{}'.format(epoch, val_metric))
            
    return train_losses, val_losses, val_metrics


def evaluate(model, loss_func, val_dl, metric=None):
    with torch.no_grad():
        results = [loss_batch(model, loss_func, anchor, image, label, metric=metric) for anchor, image, label in val_dl]
        
        losses, nums, metrics = zip(*results)
        total = np.sum(nums)
        
        avg_loss = np.sum(np.multiply(losses, nums)) / total
        
        avg_metric = None
        if metric is not None:
            avg_metric = np.sum(np.multiply(metrics, nums)) / total
            
        return avg_loss, total, avg_metric

def main():

    continue_training = False

    if len(sys.argv) == 2:
        train_checkpoint = sys.argv[1]
        continue_training = True

    train_filepath = r"/content/OP_ReID_GPTEAM/Datasets/Processed/train_ann_100.txt"
    train_img_dir = r"/content/OP_ReID_GPTEAM/Datasets/Processed/train_images"
    val_filepath = r"/content/OP_ReID_GPTEAM/Datasets/Processed/val_ann_100.txt"
    val_img_dir = r"/content/OP_ReID_GPTEAM/Datasets/Processed/val_images"
    train_bs = 64
    test_bs = 16
    num_epochs = 20
    lr = 5e4
    
    torch.autograd.set_detect_anomaly(True)
    
    transform = Transforms()
    transform_val = Transforms_val()
    train_dataset = PlushieTrainDataset(filepath=train_filepath, img_dir=train_img_dir, transform=transform)
    valid_dataset = PlushieTrainDataset(filepath=val_filepath, img_dir=val_img_dir, transform=transform_val)
    network = SiameseNetwork()
    

    print("The length of Train set is {}".format(len(train_dataset)))
    print("The length of Valid set is {}".format(len(valid_dataset)))

    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=2)
    val_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=test_bs, shuffle=True, num_workers=2)

    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    to_device(network, device)
  
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam

    if continue_training:
        checkpoint = torch.load(train_checkpoint)
        train_losses, val_losses, val_metrics = fit(num_epochs, network, criterion, 
                                            train_dl, val_dl, optimizer, lr, accuracy, checkpoint=checkpoint)
    else:
        train_losses, val_losses, val_metrics = fit(num_epochs, network, criterion, 
                                            train_dl, val_dl, optimizer, lr, accuracy)


if __name__ == "__main__":
    main()
