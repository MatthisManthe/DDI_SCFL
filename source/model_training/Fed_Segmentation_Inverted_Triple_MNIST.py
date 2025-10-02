import torch
import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import copy
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
from data.partitioning_utils import Inverted_Triple_MNIST_iid, Inverted_Triple_MNIST_Full_non_iid, Inverted_Triple_MNIST_Dirichlet_non_iid, \
    Inverted_Triple_MNIST_Fraction_non_iid, assert_Inverted_Triple_MNIST_partitioning
from monai.networks.nets import DynUNet
from metrics.metrics import CumulativeIoUTripleMNISTSegmentation
import monai
import models.monai_convolution_dropout_dim
from models.unet import UNet
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

    dataset_size = len(train_images)//2
    clients = [str(c) for c in list(range(config["nb_clients"]))]
    
    train_data = TripleMNISTSegmentationDataset(train_images, train_masks)
    val_data = TripleMNISTSegmentationDataset(val_images, val_masks)
    test_data = TripleMNISTSegmentationDataset(test_images, test_masks)
    
    if config["generate_partitioning"]:
        if config["iid"]:
            if not os.path.exists(os.path.join(config["dataset"], "IID_partitioning")):
                os.mkdir(os.path.join(config["dataset"], "IID_partitioning"))
            local_samples = Inverted_Triple_MNIST_iid(dataset_size, clients, save_folder=os.path.join(config["dataset"], "IID_partitioning"), seed=0)
            
        else:
            if config["fraction"] is not None:
                if not os.path.exists(os.path.join(config["dataset"], f"Fraction_non_IID_partitioning_{config['fraction']}")):
                    os.mkdir(os.path.join(config["dataset"], f"Fraction_non_IID_partitioning_{config['fraction']}"))
                local_samples = Inverted_Triple_MNIST_Fraction_non_iid(dataset_size, clients, config["fraction"], save_folder=os.path.join(config["dataset"], f"Fraction_non_IID_partitioning_{config['fraction']}"), seed=1)
                
            elif config["dirichlet"] == 0:
                if not os.path.exists(os.path.join(config["dataset"], "Full_non_IID_partitioning")):
                    os.mkdir(os.path.join(config["dataset"], "Full_non_IID_partitioning"))
                local_samples = Inverted_Triple_MNIST_Full_non_iid(dataset_size, clients, save_folder=os.path.join(config["dataset"], "Full_non_IID_partitioning"), seed=1)
                
            else:
                # dir 0.25: seed 6 ok, seed 5 less homogeneous but more clustered, seed 20 best so far
                # dir 0.4: seed 20 ok but ùaybe to homogeneous, seed 18 ok
                if not os.path.exists(os.path.join(config["dataset"], "Dirichlet_non_IID_partitioning")):
                    os.mkdir(os.path.join(config["dataset"], "Dirichlet_non_IID_partitioning"))
                local_samples = Inverted_Triple_MNIST_Dirichlet_non_iid(dataset_size, clients, config["dirichlet"], save_folder=os.path.join(config["dataset"], "Dirichlet_non_IID_partitioning"), seed=20)
    else:
        with open(os.path.join(config["partitioning_file"]), 'r') as f:
            local_samples = json.load(f)
        
    clients = list(local_samples.keys())
    
    # Assure that we use a correct partitioning of the data
    assert_Inverted_Triple_MNIST_partitioning(local_samples, clients)
    
    local_datasets = {} 
    for c in clients:
        full_local_samples = []
        if "Clear" in local_samples[c]:
            full_local_samples.extend(local_samples[c]["Clear"])
        if "Inverted" in  local_samples[c]:
            full_local_samples.extend([ids+dataset_size for ids in local_samples[c]["Inverted"]])
        local_datasets[c] = torch.utils.data.Subset(train_data, full_local_samples)
        
    local_loaders = {c:torch.utils.data.DataLoader(local_datasets[c], batch_size=config["batch_size"], shuffle=True, num_workers=0) for c in clients}

    print(train_data)
    print(val_data)
    print(len(train_data))
    print(len(val_data))
    plt.imshow(train_data[0][0][0], cmap="gray")
    im = plt.imshow(train_data[0][1], cmap="hot", alpha=0.3)
    plt.colorbar(im)
    plt.show()
    plt.imshow(train_data[len(train_data)//2+100][0][0], cmap="gray")
    im = plt.imshow(train_data[len(train_data)//2+100][1], cmap="hot", alpha=0.3)
    plt.colorbar(im)
    plt.show()
    
    print(train_data[len(train_data)//2+100][0].min(), train_data[len(train_data)//2+100][0].max())
    print(train_data[0][0].min(), train_data[0][0].max())
    
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    print(next(iter(val_loader))[0].shape, next(iter(val_loader))[1].shape)
    
    best_val_perfs = []
    test_perfs = []
    
    if len(config["digits"]) > 0:
        classes = ["Background"]+[str(d) for d in config["digits"]]
        nb_classes = len(config["digits"])+1
    else:
        classes = ["Background"]+list("0123456789")
        nb_classes = 11
        
    for restart in range(config["nb_restart"]):
        
        model = FCN_Triple_MNIST_Segmentation(nb_classes=nb_classes).to(device)
        
        local_models = {c:FCN_Triple_MNIST_Segmentation(nb_classes=nb_classes).to(device) for c in clients}
        
        print(model)
        print(np.sum([param.data.numel() for param in model.parameters()]))
        print(summary(model, (1,64,96)))
        
        weights = torch.tensor([0.01] + [0.99/(nb_classes-1)]*(nb_classes-1)).to(device)
        loss_func = nn.CrossEntropyLoss(weight=weights)
        
        local_optimizers = {c:optim.SGD(local_models[c].parameters(), lr = config["learning_rate"], momentum=config["momentum"]) for c in clients}
        best_metric = 0
        
        metric = CumulativeIoUTripleMNISTSegmentation(num_classes=nb_classes, classes=classes)
        
        if config["lr_scheduler"] == "exp_decay":
            local_schedulers = {c:torch.optim.lr_scheduler.ExponentialLR(local_optimizers[c], config["gamma"], verbose=True) for c in clients}
            
        for comm in range(config["max_rounds"]):  # loop over the dataset multiple times
            
            for idc, c in enumerate(clients):
                
                for local_param, global_param in zip(local_models[c].parameters(), model.parameters()):
                    local_param.data = global_param.data.clone()
                    
                local_models[c].train()
                
                # Train the model
                epoch_loss = 0.0
                
                for i, (images, labels) in enumerate(tqdm(local_loaders[c])):
        
                    output = local_models[c](images.to(device))     
                    loss = loss_func(output, labels.to(device))
                    
                    # clear gradients for this training step   
                    local_optimizers[c].zero_grad()           
                    
                    # backpropagation, compute gradients 
                    loss.backward()               
                    # apply gradients             
                    local_optimizers[c].step()                
                    epoch_loss += loss.item()*len(images)
                    
                    writer.add_scalar(f"Loss/train/Restart_{restart}_Client_{c}_per_iteration", loss.item(), comm*len(local_loaders[c])+i+1)
                    
                epoch_loss /= len(local_datasets[c])    
                
                print ('\nClient [{}/{}], Comm [{},{}], Loss: {:.4f}'.format(idc, len(clients), comm + 1, config["max_rounds"], epoch_loss))
                writer.add_scalar(f"Loss/train/Restart_{restart}_Client {c}", epoch_loss, comm+1)
            
            # Très gros con juste en dessous 
            # assert np.all([torch.equal(init, param.data) for init, param in zip(initial_state.values(), model.parameters())])
            if config["lr_scheduler"] == "exp_decay":
                for c in clients:
                    local_schedulers[c].step()
                
            for param in model.parameters():
                param.data = torch.zeros_like(param.data)
            
            for c in clients:  
                for global_param, client_param in zip(model.parameters(), local_models[c].parameters()):
                    global_param.data += client_param.data.clone() * len(local_datasets[c])/len(train_data) 
                    
            # Test the model
            model.eval()
            
            with torch.no_grad():
    
                for idl, (images, labels) in enumerate(tqdm(val_loader)):
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    test_output = model(images)
                    _, pred_y = torch.max(test_output, 1, keepdim=True)

                    metric(y_pred=pred_y, y=labels)
                    
                report = metric.aggregate()
                metric.reset()

                print(report)
                
                for cl in classes:
                    writer.add_scalar(f"Validation/Restart_{restart}_IoU_{cl}", report["IoUs"][cl], comm + 1)
                writer.add_scalar(f"Validation/Restart_{restart}_mIoU", report["mIoU"], comm + 1)
                
                if report["mIoU"] >= best_metric:
                    best_metric = report["mIoU"]
                    print("Saving new best model")
                    torch.save(model.state_dict(), os.path.join(log_dir, f"model_restart_{restart}.pth"))
                    
                    with open(os.path.join(log_dir, f'val_perf_restart_{restart}.json'), 'w') as f:
                        json.dump(report, f, indent=4) 
                
                
                if comm+1 in config["save_rounds"]:
                    shutil.copy(os.path.join(log_dir, f"model_restart_{restart}.pth"), os.path.join(log_dir, f"model_restart_{restart}_round_{comm+1}.pth"))

        best_val_perfs.append(best_metric)

        # Test the model
        model.load_state_dict(torch.load(os.path.join(log_dir, f"model_restart_{restart}.pth")))
        model.eval()
        
        with torch.no_grad():

            for idl, (images, labels) in enumerate(tqdm(test_loader)):
                images = images.to(device)
                labels = labels.to(device)
                
                test_output = model(images)
                _, pred_y = torch.max(test_output, 1, keepdim=True)

                metric(y_pred=pred_y, y=labels)

            report = metric.aggregate()
            metric.reset()

            print(report)
            
            test_perfs.append(report["mIoU"])
    
    df_metric = pd.DataFrame(data={"validation_mIoU":best_val_perfs, "test_mIoU":test_perfs})
    df_metric.to_csv(os.path.join(log_dir, "Final_results.csv"))
    

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