import torch
import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import shutil
import copy
from torch.utils.tensorboard import SummaryWriter
from time import gmtime, strftime
from torchsummary import summary
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import torch.nn as nn
from torch import optim
from sklearn.cluster import KMeans
from torch.nn.functional import normalize
import pickle 

from sklearn.metrics import classification_report, multilabel_confusion_matrix
from data.datasets import TripleMNISTSegmentationDataset
from monai.networks.nets import DynUNet
from metrics.metrics import CumulativeIoUTripleMNISTSegmentation
import monai
import models.monai_convolution_dropout_dim
from models.enet import Enet
from models.unet import UNet
from models.sinet import SINet
from models.bisenet import BiSeNetV1
from models.my_seg_net import Opt_UNet
from models.models import FCN_Triple_MNIST_Segmentation

    
def flatten(params):
    return torch.cat([param.flatten() for param in params])


def invert_full_dataset(dataset):
    return 255 - dataset
    
    
def main(args, config):
    """Main function"""
    
    # Initializing tensorboard summary
    log_dir = config["log_dir"] + "/" + os.path.basename(__file__)
    if config["log_filename"] is not None:
        log_dir += "/"+config["log_filename"]
    else:
        log_dir += "/"+strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    writer = SummaryWriter(log_dir=log_dir)
    
    # Copy config file in the experiment folder
    shutil.copy(args.config_path, log_dir)
    device = config["device"]

    train_images = np.load(os.path.join(config["dataset"], "train", config["images_name"]))
    train_masks = np.load(os.path.join(config["dataset"], "train", "masks.npy"))        
    val_images = np.load(os.path.join(config["dataset"], "val", config["images_name"]))
    val_masks = np.load(os.path.join(config["dataset"], "val", "masks.npy"))
    test_images = np.load(os.path.join(config["dataset"], "test", config["images_name"]))
    test_masks = np.load(os.path.join(config["dataset"], "test", "masks.npy")) 
  
    if config["flip_horizontal"]:
        # Half of the images of the datasets are flipped
        train_images[len(train_images)//2:] = np.flip(train_images[len(train_images)//2:], axis=-1)
        train_masks[len(train_masks)//2:] = np.flip(train_masks[len(train_masks)//2:], axis=-1)
        val_images[len(val_images)//2:] = np.flip(val_images[len(val_images)//2:], axis=-1)
        val_masks[len(val_masks)//2:] = np.flip(val_masks[len(val_masks)//2:], axis=-1)
        test_images[len(test_images)//2:] = np.flip(test_images[len(test_images)//2:], axis=-1)
        test_masks[len(test_masks)//2:] = np.flip(test_masks[len(test_masks)//2:], axis=-1)
        
    if config["flip_vertical"]:
        # Half of the images of the datasets are flipped
        train_images[len(train_images)//2:] = np.flip(train_images[len(train_images)//2:], axis=-2)
        train_masks[len(train_masks)//2:] = np.flip(train_masks[len(train_masks)//2:], axis=-2)
        val_images[len(val_images)//2:] = np.flip(val_images[len(val_images)//2:], axis=-2)
        val_masks[len(val_masks)//2:] = np.flip(val_masks[len(val_masks)//2:], axis=-2)
        test_images[len(test_images)//2:] = np.flip(test_images[len(test_images)//2:], axis=-2)
        test_masks[len(test_masks)//2:] = np.flip(test_masks[len(test_masks)//2:], axis=-2)
        
    if config["flip_90"]:
        train_images = np.pad(train_images, ((0,0), (16, 16), (0,0)), constant_values=0)
        train_masks = np.pad(train_masks, ((0,0), (16, 16), (0,0)), constant_values=0)
        val_images = np.pad(val_images, ((0,0), (16, 16), (0,0)), constant_values=0)
        val_masks = np.pad(val_masks, ((0,0), (16, 16), (0,0)), constant_values=0)
        test_images = np.pad(test_images, ((0,0), (16, 16), (0,0)), constant_values=0)
        test_masks = np.pad(test_masks, ((0,0), (16, 16), (0,0)), constant_values=0)
        
        train_images[len(train_images)//2:] = np.rot90(train_images[len(train_images)//2:], axes=(-2, -1))
        train_masks[len(train_masks)//2:] = np.rot90(train_masks[len(train_masks)//2:], axes=(-2, -1))
        val_images[len(val_images)//2:] = np.rot90(val_images[len(val_images)//2:], axes=(-2, -1))
        val_masks[len(val_masks)//2:] = np.rot90(val_masks[len(val_masks)//2:], axes=(-2, -1))
        test_images[len(test_images)//2:] = np.rot90(test_images[len(test_images)//2:], axes=(-2, -1))
        test_masks[len(test_masks)//2:] = np.rot90(test_masks[len(test_masks)//2:], axes=(-2, -1))
        
    if config["invert"]:
        # Half of the images of the datasets are inverted
        train_images[len(train_images)//2:] = invert_full_dataset(train_images[len(train_images)//2:])
        val_images[len(val_images)//2:] = invert_full_dataset(val_images[len(val_images)//2:])
        test_images[len(test_images)//2:] = invert_full_dataset(test_images[len(test_images)//2:])
         
    print(train_images.shape, train_masks.shape, val_images.shape, val_masks.shape)
    
    train_data = TripleMNISTSegmentationDataset(train_images, train_masks)
    val_data = TripleMNISTSegmentationDataset(val_images, val_masks)
    test_data = TripleMNISTSegmentationDataset(test_images, test_masks)
    
    
    print(train_data)
    print(val_data)
    print(len(train_data))
    print(len(val_data))
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs[0].imshow(train_data[0][0][0], cmap="gray")
    im = axs[1].imshow(train_data[0][1], cmap="hot", alpha=0.3)
    plt.colorbar(im)
    plt.show()
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs[0].imshow(train_data[len(train_data)//2+100][0][0], cmap="gray")
    im = axs[1].imshow(train_data[len(train_data)//2+100][1], cmap="hot", alpha=0.3)
    plt.colorbar(im)
    plt.show()
    
    print(train_data[len(train_data)//2+100][0].min(), train_data[len(train_data)//2+100][0].max())
    print(train_data[0][0].min(), train_data[0][0].max())
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    print(next(iter(train_loader))[0].shape, next(iter(train_loader))[1].shape)
    
    if len(config["digits"]) > 0:
        classes = ["Background"]+[str(d) for d in config["digits"]]
        nb_classes = len(config["digits"])+1
    else:
        classes = ["Background"]+list("0123456789")
        nb_classes = 11
        
    if config["model"] == "UNet":
        # Change the Convolution layer so that the default value of dropout dimension is 2, not well done in DynUnet.
        monai.networks.blocks.convolutions.Convolution = models.monai_convolution_dropout_dim.Convolution
        model = DynUNet(
            spatial_dims = 2,
            in_channels = 1,
            out_channels = 11,
            kernel_size = config["kernel_sizes"],
            filters = config["filters"],
            strides = config["strides"],
            upsample_kernel_size = config["strides"][1:],
            dropout=('dropout', {"p":config["dropout"]})
        ).to(device)
        
    elif config["model"] == "My_UNet":
        model = UNet(
            spatial_dims = 2,
            in_channels = 1,
            out_channels = 11,
            kernel_size = config["kernel_sizes"],
            features = config["filters"],
            strides = config["strides"],
            dropout=config["dropout"]
        ).to(device)
        
    elif config["model"] == "ENet":
        model = Enet(
            input_shape=(1,64,96),
            output_shape=(11,64,96)
        ).to(device)
        
    elif config["model"] == "SInet":
        model = SINet()
        
    elif config["model"] == "bisenet":
        model = BiSeNetV1(11, in_channels=1, aux_mode='train').to(device)
        
    elif config["model"] == "my_opt_UNet":
        model = Opt_UNet(
            spatial_dims = 2,
            in_channels = 1,
            out_channels = 11,
            kernel_size = config["kernel_sizes"],
            features = config["filters"],
            strides = config["strides"],
            dropout=config["dropout"]).to(device)
        
    elif config["model"] == "My_fcn":
        model = FCN_Triple_MNIST_Segmentation(nb_classes=nb_classes).to(device)
        
    #print(unet)
    #print(summary(model, (1,64,96)))
    print(model)
    print(np.sum([param.data.numel() for param in model.parameters()]))
    
    weights = torch.tensor([0.01] + [0.99/(nb_classes-1)]*(nb_classes-1)).to(device)
    loss_func = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = optim.SGD(model.parameters(), lr = config["learning_rate"], momentum=config["momentum"])

    if config["lr_scheduler"] == "exp_decay":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, config["gamma"], verbose=True) 
        
    model.train()
        
    # Train the model
    total_step = len(train_loader)
    best_val = 0
    
    metric = CumulativeIoUTripleMNISTSegmentation(num_classes=nb_classes, classes=classes)
    
    for epoch in range(config["nb_epoch"]):
        
        epoch_loss = 0.0
        
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            
            if config["model"] == "bisenet":
                output, _, _ = model(images.to(device)) 
            else:
                output = model(images.to(device)) 
                
            loss = loss_func(output, labels.to(device))
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()               
            # apply gradients             
            optimizer.step()    

            epoch_loss += loss.item()*len(images)            
            
            if config["lr_scheduler"] == "exp_decay":
                scheduler.step()
                
        epoch_loss /= len(train_data) 
        
        print ('\nEpoch [{},{}], Loss: {:.4f}'.format(epoch + 1, config["nb_epoch"], epoch_loss))
        writer.add_scalar("Loss/train", epoch_loss, epoch+1)
    
        if config["lr_scheduler"] == "exp_decay":
            scheduler.step()
            
        # Test the model
        model.eval()    
        
        with torch.no_grad():

            for idl, (images, labels) in enumerate(tqdm(val_loader)):
                images = images.to(device)
                labels = labels.to(device)
                
                if config["model"] == "bisenet":
                    test_output, _, _ = model(images)
                else:
                    test_output = model(images)
                _, pred_y = torch.max(test_output, 1, keepdim=True)
                
                """
                if idl < 50 or idl > 320 and idl < 370:
                    fig, axs = plt.subplots(3,1)
                    axs[0].imshow(images[0][0].cpu(), cmap="gray")
                    im = axs[1].imshow(labels[0].cpu(), cmap="hot", vmin=0, vmax=nb_classes-1)
                    plt.colorbar(im)
                    im = axs[2].imshow(pred_y[0][0].cpu(), cmap="hot", vmin=0, vmax=nb_classes-1)
                    plt.colorbar(im)
                    plt.show()
                """
                
                metric(y_pred=pred_y, y=labels)
                    
            report = metric.aggregate()
            metric.reset()
            
            for c in classes:
                writer.add_scalar(f"Validation/IoU_{c}", report["IoUs"][c], epoch + 1)
            writer.add_scalar("Validation/mIoU", report["mIoU"], epoch + 1)
            
            print(report)
            
            if best_val < report["mIoU"]:
                best_val = report["mIoU"]
                torch.save(model.state_dict(), os.path.join(log_dir, "model.pth"))
                print("\nSaved new best model")
            
        
    if config["test"]:
        
        model.load_state_dict(torch.load(os.path.join(log_dir, "model.pth")))
        
        # Final test 
        model.eval()    
        
        with torch.no_grad():

            for idl, (images, labels) in enumerate(tqdm(test_loader)):
                images = images.to(device)
                labels = labels.to(device)
                
                test_output = model(images)
                _, pred_y = torch.max(test_output, 1, keepdim=True)
                
                metric(y_pred=pred_y, y=labels)

                if idl < 3 or (idl > len(test_loader)//2+1 and idl < len(test_loader)//2+4):
                    for b in range(min(5, len(images))):
                        fig, axs = plt.subplots(3,1)
                        axs[0].imshow(images[b][0].cpu(), cmap="gray")
                        im = axs[1].imshow(labels[b].cpu(), cmap="hot", vmin=0, vmax=nb_classes-1)
                        plt.colorbar(im)
                        im = axs[2].imshow(pred_y[b][0].cpu(), cmap="hot", vmin=0, vmax=nb_classes-1)
                        plt.colorbar(im)
                        plt.show()
                    
            report = metric.aggregate()

            with open(os.path.join(log_dir, 'test_perf.json'), 'w') as f:
                json.dump(report, f)           
    
    

if __name__ == "__main__":
    
    print("Curent working directory: ", os.getcwd())
    
    print("Is cuda avaiable? ", torch.cuda.is_available())
    print("Number of cuda devices available: ", torch.cuda.device_count())
    
    # Define argument parser and its attributes
    parser = argparse.ArgumentParser(description='Train 3D UNet on Brats')
    
    parser.add_argument('--config_path', dest='config_path', type=str,
                        help='path to json config file')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Read the config file
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    main(args, config)